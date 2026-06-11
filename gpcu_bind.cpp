// gpcu_bind.cpp -- pybind11 bindings for the CUDA GP.
//
// API is intentionally identical to gp_bind.cpp so benchmark.py can
// call both backends with the same code paths.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "gp_cuda.h"
#include "gp_cuda_kernel.h"

namespace py = pybind11;
using arr_d = py::array_t<float, py::array::c_style | py::array::forcecast>;

class PyGPCU {
    GPCU *gp_;

    void check() const {
        if (!gp_) throw std::runtime_error("GPCU object is invalid");
    }

    static int check_X(py::buffer_info &buf, int dim) {
        if (buf.ndim == 1) {
            if (dim != 1)
                throw std::invalid_argument("1-D X requires dim=1");
            return (int)buf.shape[0];
        }
        if (buf.ndim == 2) {
            if ((int)buf.shape[1] != dim)
                throw std::invalid_argument(
                    "X has " + std::to_string(buf.shape[1]) +
                    " cols, expected " + std::to_string(dim));
            return (int)buf.shape[0];
        }
        throw std::invalid_argument("X must be 1-D or 2-D");
    }

public:
    PyGPCU(int dim, int capacity,
           float lengthscale, float outputscale, float noise, float offset)
        : gp_(gpcu_create(dim, capacity,
                          gpcu_kernel_matern32_linear(dim, lengthscale, outputscale, offset),
                          noise))
    {
        if (!gp_) throw std::runtime_error("gpcu_create failed");
    }

    explicit PyGPCU(GPCU *raw) : gp_(raw) {}
    ~PyGPCU() { if (gp_) gpcu_destroy(gp_); }

    PyGPCU(const PyGPCU &)            = delete;
    PyGPCU &operator=(const PyGPCU &) = delete;
    PyGPCU(PyGPCU &&o) noexcept : gp_(o.gp_) { o.gp_ = nullptr; }
    PyGPCU &operator=(PyGPCU &&o) noexcept {
        if (gp_) gpcu_destroy(gp_);
        gp_ = o.gp_; o.gp_ = nullptr;
        return *this;
    }

    // training

    void fit(arr_d X, arr_d y) {
        check();
        auto xbuf = X.request(), ybuf = y.request();
        int n = check_X(xbuf, gp_->dim);
        if ((int)ybuf.shape[0] != n)
            throw std::invalid_argument("X and y length mismatch");
        int rc = gpcu_fit(gp_, (const float *)xbuf.ptr,
                               (const float *)ybuf.ptr, n, 0);
        if (rc == -1) throw std::runtime_error("n exceeds capacity");
        if (rc == -2) throw std::runtime_error("Cholesky factorisation failed");
    }

    void recompute() { check(); gpcu_recompute(gp_, 0); }

    // prediction

    py::tuple predict(arr_d X, bool noise = false) {
        check();
        auto buf = X.request();
        int m = check_X(buf, gp_->dim);

        auto means = py::array_t<float>(m);
        auto vars  = py::array_t<float>(m);
        gpcu_predict(gp_, (const float *)buf.ptr,
                     (float *)means.request().ptr,
                     (float *)vars.request().ptr, m, 0);
        if (noise) {
            float sn = gpcu_get_noise(gp_);
            float *vp = (float *)vars.request().ptr;
            for (int i = 0; i < m; i++) vp[i] += sn;
        }
        return py::make_tuple(means, vars);
    }

    // log marginal likelihood + gradients

    float log_marginal_likelihood() const {
        check(); return gpcu_marginal_log_likelihood(gp_);
    }

    // Returns flat tuple (ell_0,...,ell_{dim-1}, sf, noise, offset) in raw param space.
    py::tuple mll_grad() const {
        check();
        int np = gp_->kernel->n_params;
        int d  = gp_->dim;
        float d_raw_noise;
        std::vector<float> kg((size_t)np);
        gpcu_mll_grad(gp_, &d_raw_noise, kg.data(), 0);
        py::list lst;
        for (int i = 0; i < d; i++)   lst.append(kg[i]);
        lst.append(kg[np - 2]);  // sf
        lst.append(d_raw_noise); // noise
        lst.append(kg[np - 1]);  // offset
        return py::tuple(lst);
    }

    // constrained hyperparameters

    py::array_t<float> lengthscale() const {
        check();
        int d = gp_->dim;
        py::array_t<float> res(d);
        auto buf = res.mutable_unchecked<1>();
        for (int i = 0; i < d; i++) buf(i) = gpcu_kernel_get_lengthscale(gp_->kernel, i);
        return res;
    }
    float outputscale() const { check(); return gpcu_kernel_get_outputscale(gp_->kernel); }
    float noise()       const { check(); return gpcu_get_noise(gp_);                      }
    float offset()      const { check(); return gpcu_kernel_get_offset(gp_->kernel);      }

    void set_lengthscale(arr_d v) {
        check();
        auto buf = v.request();
        if ((int)buf.shape[0] != gp_->dim)
            throw std::invalid_argument(
                "wrong lengthscale size: expected " + std::to_string(gp_->dim));
        const float *p = (const float *)buf.ptr;
        for (int i = 0; i < gp_->dim; i++) gpcu_kernel_set_lengthscale(gp_->kernel, i, p[i]);
    }
    void set_outputscale(float v) { check(); gpcu_kernel_set_outputscale(gp_->kernel, v); }
    void set_noise      (float v) { check(); gpcu_set_noise(gp_, v);                      }
    void set_offset     (float v) { check(); gpcu_kernel_set_offset(gp_->kernel, v);      }

    // raw (unconstrained) hyperparameters

    py::array_t<float> raw_lengthscale() const {
        check();
        int d = gp_->dim;
        py::array_t<float> res(d);
        auto buf = res.mutable_unchecked<1>();
        for (int i = 0; i < d; i++) buf(i) = gp_->kernel->raw_params[i];
        return res;
    }
    float raw_outputscale() const { check(); return gp_->kernel->raw_params[gp_->kernel->n_params - 2]; }
    float raw_noise()       const { check(); return gp_->raw_noise; }
    float raw_offset()      const { check(); return gp_->kernel->raw_params[gp_->kernel->n_params - 1]; }

    void set_raw_lengthscale(arr_d v) {
        check();
        auto buf = v.request();
        if ((int)buf.shape[0] != gp_->dim)
            throw std::invalid_argument(
                "wrong lengthscale size: expected " + std::to_string(gp_->dim));
        const float *p = (const float *)buf.ptr;
        for (int i = 0; i < gp_->dim; i++) gp_->kernel->raw_params[i] = p[i];
    }
    void set_raw_outputscale(float v) { check(); gp_->kernel->raw_params[gp_->kernel->n_params - 2] = v; }
    void set_raw_noise      (float v) { check(); gp_->raw_noise = v; }
    void set_raw_offset     (float v) { check(); gp_->kernel->raw_params[gp_->kernel->n_params - 1] = v; }

    // misc

    int    n()               const { check(); return gp_->n;                }
    int    dim()             const { check(); return gp_->dim;              }
    int    capacity()        const { check(); return gp_->cap;              }
    float dedup_threshold() const { check(); return gp_->dedup_threshold;  }
    void set_dedup_threshold(float v) { check(); gp_->dedup_threshold = v; }

    void save(const std::string &path) const {
        check();
        if (gpcu_save(gp_, path.c_str()) != 0)
            throw std::runtime_error("save failed: " + path);
    }

    static PyGPCU load(const std::string &path, int extra_cap = 0) {
        GPCU *raw = gpcu_load(path.c_str(), extra_cap);
        if (!raw) throw std::runtime_error("load failed: " + path);
        return PyGPCU(raw);
    }
};

// module

PYBIND11_MODULE(puffing_gpcu, m)
{
    m.doc() = "GP regression -- CUDA (cuBLAS/cuSOLVER + pybind11)";

    py::class_<PyGPCU>(m, "GP")
        .def(py::init<int, int, float, float, float, float>(),
             py::arg("dim"), py::arg("capacity"),
             py::arg("lengthscale") = 1.0,
             py::arg("outputscale") = 1.0,
             py::arg("noise")       = 1e-2,
             py::arg("offset")      = 1.0)

        .def("fit",       &PyGPCU::fit,       py::arg("X"), py::arg("y"))
        .def("recompute", &PyGPCU::recompute)
        .def("predict",   &PyGPCU::predict,   py::arg("X"), py::arg("noise") = false)
        .def("mll_grad",  &PyGPCU::mll_grad,
             "Returns flat tuple (ell_0,...,ell_{dim-1}, sf, noise, offset) in raw param space")

        .def_property("lengthscale", &PyGPCU::lengthscale, &PyGPCU::set_lengthscale)
        .def_property("outputscale", &PyGPCU::outputscale, &PyGPCU::set_outputscale)
        .def_property("noise",       &PyGPCU::noise,       &PyGPCU::set_noise)
        .def_property("offset",      &PyGPCU::offset,      &PyGPCU::set_offset)

        .def_property("raw_lengthscale",
                      &PyGPCU::raw_lengthscale, &PyGPCU::set_raw_lengthscale)
        .def_property("raw_outputscale",
                      &PyGPCU::raw_outputscale, &PyGPCU::set_raw_outputscale)
        .def_property("raw_noise",
                      &PyGPCU::raw_noise,       &PyGPCU::set_raw_noise)
        .def_property("raw_offset",
                      &PyGPCU::raw_offset,      &PyGPCU::set_raw_offset)

        .def_property_readonly("log_marginal_likelihood",
                               &PyGPCU::log_marginal_likelihood)
        .def_property_readonly("n",        &PyGPCU::n)
        .def_property_readonly("dim",      &PyGPCU::dim)
        .def_property_readonly("capacity", &PyGPCU::capacity)
        .def_property("dedup_threshold",
                      &PyGPCU::dedup_threshold, &PyGPCU::set_dedup_threshold)

        .def("save",        &PyGPCU::save, py::arg("path"))
        .def_static("load", &PyGPCU::load,
                    py::arg("path"), py::arg("extra_cap") = 0)

        .def("__repr__", [](const PyGPCU &g) {
            return "<GP dim=" + std::to_string(g.dim()) +
                   " n="   + std::to_string(g.n()) +
                   " cap=" + std::to_string(g.capacity()) + ">";
        });
}

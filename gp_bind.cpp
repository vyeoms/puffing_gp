// gp_bind.cpp -- pybind11 bindings for the CBLAS/LAPACK GP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "gp.h"
#include "gp_kernel.h"

namespace py = pybind11;
using arr_d = py::array_t<double, py::array::c_style | py::array::forcecast>;

class PyGP {
    GP *gp_;

    void check() const {
        if (!gp_) throw std::runtime_error("GP object is invalid");
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
    PyGP(int dim, int capacity,
         double lengthscale, double outputscale, double noise, double offset)
        : gp_(gp_create(dim, capacity,
                        gp_kernel_matern32_linear(dim, lengthscale, outputscale, offset),
                        noise))
    {
        if (!gp_) throw std::runtime_error("gp_create failed");
    }

    explicit PyGP(GP *raw) : gp_(raw) {}
    ~PyGP() { if (gp_) gp_destroy(gp_); }

    PyGP(const PyGP &)            = delete;
    PyGP &operator=(const PyGP &) = delete;
    PyGP(PyGP &&o) noexcept : gp_(o.gp_) { o.gp_ = nullptr; }
    PyGP &operator=(PyGP &&o) noexcept {
        if (gp_) gp_destroy(gp_);
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
        int rc = gp_fit(gp_, (const double *)xbuf.ptr,
                             (const double *)ybuf.ptr, n);
        if (rc == -1) throw std::runtime_error("n exceeds capacity");
        if (rc == -2) throw std::runtime_error("Cholesky factorisation failed");
    }

    void recompute() { check(); gp_recompute(gp_); }

    // prediction

    py::tuple predict(arr_d X, bool noise = false) {
        check();
        auto buf = X.request();
        int m = check_X(buf, gp_->dim);

        auto means = py::array_t<double>(m);
        auto vars  = py::array_t<double>(m);
        gp_predict(gp_, (const double *)buf.ptr,
                   (double *)means.request().ptr,
                   (double *)vars.request().ptr, m);
        if (noise) {
            double sn = gp_get_noise(gp_);
            double *vp = (double *)vars.request().ptr;
            for (int i = 0; i < m; i++) vp[i] += sn;
        }
        return py::make_tuple(means, vars);
    }

    // log marginal likelihood + gradients

    double log_marginal_likelihood() const {
        check(); return gp_marginal_log_likelihood(gp_);
    }

    // Returns flat tuple (ell_0,...,ell_{dim-1}, sf, noise, offset) in raw param space.
    py::tuple mll_grad() const {
        check();
        int np = gp_->kernel->n_params;
        int d  = gp_->dim;
        double d_raw_noise;
        std::vector<double> kg((size_t)np);
        gp_mll_grad(gp_, &d_raw_noise, kg.data());
        // kernel layout: [0..d-1]=ell, [np-2]=sf, [np-1]=offset
        py::list lst;
        for (int i = 0; i < d; i++)   lst.append(kg[i]);
        lst.append(kg[np - 2]);  // sf
        lst.append(d_raw_noise); // noise
        lst.append(kg[np - 1]);  // offset
        return py::tuple(lst);
    }

    // constrained hyperparameters (arrays for lengthscale, scalars for rest)

    py::array_t<double> lengthscale() const {
        check();
        int d = gp_->dim;
        py::array_t<double> res(d);
        auto buf = res.mutable_unchecked<1>();
        for (int i = 0; i < d; i++) buf(i) = gp_kernel_get_lengthscale(gp_->kernel, i);
        return res;
    }
    double outputscale() const { check(); return gp_kernel_get_outputscale(gp_->kernel); }
    double noise()       const { check(); return gp_get_noise(gp_);                      }
    double offset()      const { check(); return gp_kernel_get_offset(gp_->kernel);      }

    void set_lengthscale(arr_d v) {
        check();
        auto buf = v.request();
        if ((int)buf.shape[0] != gp_->dim)
            throw std::invalid_argument(
                "wrong lengthscale size: expected " + std::to_string(gp_->dim));
        const double *p = (const double *)buf.ptr;
        for (int i = 0; i < gp_->dim; i++) gp_kernel_set_lengthscale(gp_->kernel, i, p[i]);
    }
    void set_outputscale(double v) { check(); gp_kernel_set_outputscale(gp_->kernel, v); }
    void set_noise      (double v) { check(); gp_set_noise(gp_, v);                      }
    void set_offset     (double v) { check(); gp_kernel_set_offset(gp_->kernel, v);      }

    // raw (unconstrained) hyperparameters

    py::array_t<double> raw_lengthscale() const {
        check();
        int d = gp_->dim;
        py::array_t<double> res(d);
        auto buf = res.mutable_unchecked<1>();
        for (int i = 0; i < d; i++) buf(i) = gp_->kernel->raw_params[i];
        return res;
    }
    double raw_outputscale() const { check(); return gp_->kernel->raw_params[gp_->kernel->n_params - 2]; }
    double raw_noise()       const { check(); return gp_->raw_noise; }
    double raw_offset()      const { check(); return gp_->kernel->raw_params[gp_->kernel->n_params - 1]; }

    void set_raw_lengthscale(arr_d v) {
        check();
        auto buf = v.request();
        if ((int)buf.shape[0] != gp_->dim)
            throw std::invalid_argument(
                "wrong lengthscale size: expected " + std::to_string(gp_->dim));
        const double *p = (const double *)buf.ptr;
        for (int i = 0; i < gp_->dim; i++) gp_->kernel->raw_params[i] = p[i];
    }
    void set_raw_outputscale(double v) { check(); gp_->kernel->raw_params[gp_->kernel->n_params - 2] = v; }
    void set_raw_noise      (double v) { check(); gp_->raw_noise = v; }
    void set_raw_offset     (double v) { check(); gp_->kernel->raw_params[gp_->kernel->n_params - 1] = v; }

    // misc

    int    n()               const { check(); return gp_->n;                }
    int    dim()             const { check(); return gp_->dim;              }
    int    capacity()        const { check(); return gp_->cap;              }
    double dedup_threshold() const { check(); return gp_->dedup_threshold;  }
    void set_dedup_threshold(double v) { check(); gp_->dedup_threshold = v; }

    void save(const std::string &path) const {
        check();
        if (gp_save(gp_, path.c_str()) != 0)
            throw std::runtime_error("save failed: " + path);
    }

    static PyGP load(const std::string &path, int extra_cap = 0) {
        GP *raw = gp_load(path.c_str(), extra_cap);
        if (!raw) throw std::runtime_error("load failed: " + path);
        return PyGP(raw);
    }
};

// module

PYBIND11_MODULE(puffing_gp, m)
{
    m.doc() = "GP regression -- CBLAS/LAPACK (C + pybind11)";

    py::class_<PyGP>(m, "GP")
        .def(py::init<int, int, double, double, double, double>(),
             py::arg("dim"), py::arg("capacity"),
             py::arg("lengthscale") = 1.0,
             py::arg("outputscale") = 1.0,
             py::arg("noise")       = 1e-2,
             py::arg("offset")      = 1.0)

        .def("fit",       &PyGP::fit,       py::arg("X"), py::arg("y"))
        .def("recompute", &PyGP::recompute)
        .def("predict",   &PyGP::predict,   py::arg("X"), py::arg("noise") = false)

        .def("mll_grad", &PyGP::mll_grad,
             "Returns flat tuple (ell_0,...,ell_{dim-1}, sf, noise, offset) in raw param space")

        .def_property("lengthscale", &PyGP::lengthscale, &PyGP::set_lengthscale)
        .def_property("outputscale", &PyGP::outputscale, &PyGP::set_outputscale)
        .def_property("noise",       &PyGP::noise,       &PyGP::set_noise)
        .def_property("offset",      &PyGP::offset,      &PyGP::set_offset)

        .def_property("raw_lengthscale",
                      &PyGP::raw_lengthscale, &PyGP::set_raw_lengthscale)
        .def_property("raw_outputscale",
                      &PyGP::raw_outputscale, &PyGP::set_raw_outputscale)
        .def_property("raw_noise",
                      &PyGP::raw_noise,       &PyGP::set_raw_noise)
        .def_property("raw_offset",
                      &PyGP::raw_offset,      &PyGP::set_raw_offset)

        .def_property_readonly("log_marginal_likelihood",
                               &PyGP::log_marginal_likelihood)
        .def_property_readonly("n",        &PyGP::n)
        .def_property_readonly("dim",      &PyGP::dim)
        .def_property_readonly("capacity", &PyGP::capacity)
        .def_property("dedup_threshold",
                      &PyGP::dedup_threshold, &PyGP::set_dedup_threshold)

        .def("save",        &PyGP::save, py::arg("path"))
        .def_static("load", &PyGP::load,
                    py::arg("path"), py::arg("extra_cap") = 0)

        .def("__repr__", [](const PyGP &g) {
            return "<GP dim=" + std::to_string(g.dim()) +
                   " n="   + std::to_string(g.n()) +
                   " cap=" + std::to_string(g.capacity()) + ">";
        });
}

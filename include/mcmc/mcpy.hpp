#ifndef MCPY_HPP
#define MCPY_HPP


#include <pybind11/attr.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include "ising.hpp"
#include "pdf.hpp"
#include <array>
#include <stdexcept>


namespace py = pybind11;

template<typename Scalar>
struct PySample;


template<typename Scalar>
struct PyBinnedSample;

template<class Scalar, class ArrayType>
py::array_t<Scalar> to_numpy(const ArrayType& array, const std::vector<size_t>& q0_shape={}){
    if (q0_shape.size() == 0){
        py::array_t<Scalar> res(array.size(), array.data());
        return res;
    }
    else{
        py::array_t<Scalar> res(q0_shape, array.data());
        return res;
    }
}


template<typename Scalar, typename ArrayType>
py::tuple to_tuple(const ArrayType& arr) {
    py::tuple py_tuple(arr.size());
    for (size_t i = 0; i < arr.size(); ++i) {
        py_tuple[i] = Scalar(arr[i]);
    }
    return py_tuple;
}

template<typename Scalar, size_t Dim>
std::vector<Scalar> flatten(const std::vector<std::array<Scalar, Dim>>& f){
    size_t nt = f.size();
    if (nt == 0){
        return {};
    }
    std::vector<Scalar> res(nt*Dim);

    for (size_t i=0; i<nt; i++){
        for (size_t j=0; j<Dim; j++){
            res[i*Dim + j] = f[i][j];
        }
    }
    return res;
}



template<typename Scalar>
std::vector<Scalar> to_vector(const py::iterable& iterable){
    std::vector<Scalar> res;
    for (const py::handle &item : iterable)
    {
        res.push_back(py::cast<Scalar>(item));
    }
    res.shrink_to_fit();
    return res;
}

template<typename State, typename PyState>
py::list to_pystates(const std::vector<State>& states){
    py::list res(states.size());
    for (size_t i=0; i<states.size(); i++){
        res[i] = PyState(states[i]);
    }
    return res;
}

template<typename Scalar>
py::list to_pystates(const std::vector<CoordsND<Scalar, 1>>& states){
    py::list res(states.size());
    for (size_t i=0; i<states.size(); i++){
        res[i] = states[i][0];
    }
    return res;
}

template<typename Scalar, size_t Dim>
py::list to_pystates(const std::vector<CoordsND<Scalar, Dim>>& states){
    py::list res(states.size());
    for (size_t i=0; i<states.size(); i++){
        res[i] = to_tuple<Scalar>(states[i]);
    }
    return res;
}

template<typename Scalar>
py::list to_pysamples(const std::vector<Sample<Scalar>>& samples){
    py::list res(samples.size());
    for (size_t i=0; i<samples.size(); i++){
        res[i] = PySample(samples[i]);
    }
    return res;
}

template<typename Scalar, typename State, typename PyState>
Observable<Scalar, State> to_observable(py::function f){
    if (typeid(State) == typeid(PyState)){
        return [f](const State& state){
            return f(&state).template cast<Scalar>();
        };
    }
    else{
        return [f](const State& state){
            return f(PyState(state)).template cast<Scalar>();
        };
    }
}

template<typename Scalar, size_t Dim>
LimitsND<Scalar, Dim> to_limits(py::iterable py_limits) {
    LimitsND<Scalar, Dim> limits;

    size_t i = 0;
    py::iterator outer_it = py_limits.begin();
    py::iterator outer_end = py_limits.end();

    while (outer_it != outer_end) {
        if (i >= Dim)
            throw std::runtime_error("Too many limits provided. Expected " + std::to_string(Dim));

        py::handle item = *outer_it;
        py::iterable pair = py::cast<py::iterable>(item);
        py::iterator inner_it = pair.begin();
        py::iterator inner_end = pair.end();

        if (inner_it == inner_end)
            throw std::runtime_error("Limit pair too short");

        Scalar a = py::cast<Scalar>(*inner_it++);
        if (inner_it == inner_end)
            throw std::runtime_error("Limit pair too short");

        Scalar b = py::cast<Scalar>(*inner_it++);
        if (inner_it != inner_end)
            throw std::runtime_error("Limit pair too long");

        limits[i][0] = a;
        limits[i][1] = b;

        ++i;
        ++outer_it;
    }

    if (i != Dim)
        throw std::runtime_error("Too few limits provided. Expected " + std::to_string(Dim));

    return limits;
}


template<typename Scalar>
std::function<Scalar(const CoordsND<Scalar, 1>&)> to_pdf(py::function f){
    return [f](const CoordsND<Scalar, 1>& x) -> Scalar {
        return f(x[0]).template cast<Scalar>();
    };
}

template<typename Scalar, size_t Dim>
std::function<Scalar(const CoordsND<Scalar, Dim>&)> to_pdf_ND(py::function f){
    return [f](const CoordsND<Scalar, Dim>& x) -> Scalar {
        return f(*to_tuple<Scalar>(x)).template cast<Scalar>();
    };
}

template<typename Scalar>
struct PySample : public Sample<Scalar>{

    PySample(const py::iterable& array) : Sample<Scalar>(to_vector<Scalar>(array)){}

    PySample(const Sample<Scalar>& sample) : Sample<Scalar>(sample){}

    PyBinnedSample<Scalar> pybin_it() const{
        return this->bin_it();
    }
};


template<typename Scalar>
struct PyBinnedSample : public BinningAnalysis<Scalar>{

    PyBinnedSample(const py::iterable& array):BinningAnalysis<Scalar>(to_vector<Scalar>(array)){}

    PyBinnedSample(const Sample<Scalar>& sample) : BinningAnalysis<Scalar>(sample){}

    PyBinnedSample(const BinningAnalysis<Scalar>& obj) : BinningAnalysis<Scalar>(obj){}
};


template<typename Scalar>
class PyMonteCarlo{

public:

    PyMonteCarlo() = default;

    virtual ~PyMonteCarlo();

    inline size_t               N() const { return mc().N();}

    void                        update(const std::string& method, const size_t& steps, const size_t& sweeps){
        mc().update(method, steps, sweeps);
    }

    void                        thermalize(const std::string& method, const size_t& sweeps){
        mc().thermalize(method, sweeps);
    }

    virtual MonteCarlo<Scalar>& mc() = 0;

    virtual PySample<Scalar>    py_sample(py::function obs) const = 0;
    
    virtual py::list            py_data() const = 0;

};


template<typename Scalar>
class PyPDF : public PyMonteCarlo<Scalar>{

public:

    PyPDF() = default;

    virtual py::array_t<Scalar> array_data() const = 0;

    virtual py::array_t<Scalar> draw(const size_t& samples, const size_t& therm_factor=0) = 0;
};


template<typename Scalar>
class PyPDF1D : public PyPDF<Scalar>{

public:
    PyPDF1D(py::function f, Scalar a, Scalar b, Scalar step) : PyPDF<Scalar>(), _mc(to_pdf<Scalar>(f), {{{a, b}}}, step){}

    MonteCarlo<Scalar>& mc() override{
        return _mc;
    }

    PySample<Scalar>    py_sample(py::function obs) const override{
        std::function<Scalar(const CoordsND<Scalar, 1>&)> func = [obs](const CoordsND<Scalar, 1>& state){
            return obs(state[0]).template cast<Scalar>();
        };
        return _mc.sample(func);
    }

    py::list            py_data() const override{
        py::list res(_mc.N());
        for (size_t i=0; i<_mc.N(); i++){
            res[i] = _mc.data()[i][0];
        }
        return res;
    }

    py::array_t<Scalar> array_data() const override{
        std::vector<Scalar> res(_mc.N());
        for (size_t i=0; i<_mc.N(); i++){
            res[i] = _mc.data()[i][0];
        }
        return to_numpy<Scalar>(res);
    }

    py::array_t<Scalar> draw(const size_t& samples, const size_t& therm_factor=0) override{
        std::vector<CoordsND<Scalar, 1>> res = _mc.draw(samples, therm_factor);
        std::vector<Scalar> py_res(samples);
        for (size_t i=0; i<samples; i++){
            py_res[i] = res[i][0];
        }
        return to_numpy<Scalar>(py_res);
    }

private:

    MonteCarloPDF<Scalar, 1> _mc;
};


template<typename Scalar, size_t Dim>
class PyPDFND : public PyPDF<Scalar>{

public:
    PyPDFND(py::function f, py::iterable limits, Scalar step) : PyPDF<Scalar>(), _mc(to_pdf_ND<Scalar, Dim>(f), to_limits<Scalar, Dim>(limits), step){}

    MonteCarlo<Scalar>& mc() override{
        return _mc;
    }

    PySample<Scalar> py_sample(py::function obs) const override{
        std::function<Scalar(const CoordsND<Scalar, Dim>&)> func = [obs](const CoordsND<Scalar, Dim>& state){
            return obs(*to_tuple<Scalar>(state)).template cast<Scalar>();
        };
        return _mc.sample(func);
    }

    py::list            py_data() const override{
        py::list res(_mc.N());
        for (size_t i=0; i<_mc.N(); i++){
            res[i] = to_tuple<Scalar>(_mc.data()[i]);
        }
        return res;
    }

    py::array_t<Scalar> array_data() const override{
        std::vector<Scalar> res = flatten<Scalar>(_mc.data());
        return to_numpy<Scalar>(res, {_mc.N(), Dim});
    }

    py::array_t<Scalar> draw(const size_t& samples, const size_t& therm_factor=0) override{
        std::vector<CoordsND<Scalar, Dim>> res = _mc.draw(samples, therm_factor);
        std::vector<Scalar> py_res = flatten<Scalar>(res);
        return to_numpy<Scalar>(py_res, {res.size(), Dim});
    }

private:

    MonteCarloPDF<Scalar, Dim> _mc;
};


template<typename Scalar>
class PyIsingModel2D : public PyMonteCarlo<Scalar>{

public:

    PyIsingModel2D(Scalar T, size_t Lx, size_t Ly) : PyMonteCarlo<Scalar>(), _model(T, Lx, Ly) {}

    MonteCarlo<Scalar>& mc() final{
        return _model;
    }

    PySample<Scalar> py_sample(py::function obs) const{
        return _model.sample(to_observable<Scalar, SpinState<Scalar>, SpinState<Scalar>>(obs));
    }

    py::list            py_data() const{
        const std::vector<SpinState<Scalar>>& res = _model.data();
        py::list pyres(res.size());
        for (size_t i=0; i<res.size(); i++){
            pyres[i] = res[i];
        }
        return pyres;
    }

    Scalar T() const{
        return _model.T();
    }

    void ssf_update(size_t steps, size_t sweeps){
        _model.ssf_update(steps, sweeps);
    }

    void wolff_update(size_t steps, size_t sweeps){
        _model.wolff_update(steps, sweeps);
    }

    void ssf_thermalize(size_t sweeps){
        _model.ssf_update(sweeps);
    }

    void wolff_thermalize(size_t sweeps){
        _model.wolff_update(sweeps);
    }

    PySample<Scalar> energy_sample() const {
        return _model.energy_sample();
    }



private:

    IsingModel2D<Scalar> _model;

};


template<typename Scalar>
void py_update_all(py::iterable obj, py::str py_method, const size_t& steps, const size_t& sweeps, int threads){
    std::string method = py_method.cast<std::string>();
    std::vector<PyMonteCarlo<Scalar>*> array;
    for (const py::handle& item : obj){
        array.push_back(&item.cast<PyMonteCarlo<Scalar>&>());
    }


    threads = (threads <= 0) ? omp_get_max_threads() : threads;
    #pragma omp parallel for num_threads(threads)
    for (size_t i=0; i<array.size(); i++){
        array[i]->update(method, steps, sweeps);
    }
}


template<typename Scalar>
void define_base_module(py::module &m){

    py::class_<PySample<Scalar>>(m, "Sample")
        .def(py::init<py::iterable>(), py::arg("data"))
        .def_property_readonly("data", [](const PySample<Scalar> &self){ return to_numpy<Scalar>(self.sample); })
        .def_property_readonly("N", &Sample<Scalar>::N)
        .def("mean", &Sample<Scalar>::mean)
        .def("std", &Sample<Scalar>::std)
        .def("popul_std", &Sample<Scalar>::popul_std)
        .def("error", &Sample<Scalar>::error)
        .def("bin_it", &PySample<Scalar>::pybin_it)
        .def_property_readonly("stat", &Sample<Scalar>::message);

    py::class_<PyBinnedSample<Scalar>>(m, "BinnedSample")
        .def(py::init<py::iterable>(), py::arg("data"))
        .def_property_readonly("max_level", &PyBinnedSample<Scalar>::max_level)
        .def_property_readonly("converged", &PyBinnedSample<Scalar>::converged)
        .def_property_readonly("tau_estimate", &PyBinnedSample<Scalar>::tau_estimate)
        .def_property_readonly("binned_samples", [](const PyBinnedSample<Scalar>& self){return to_pysamples(self.samples());});
    
    
    py::class_<PyMonteCarlo<Scalar>, std::unique_ptr<PyMonteCarlo<Scalar>>>(m, "MonteCarlo")
        .def_property_readonly("N", [](const MonteCarlo<Scalar>& self){return self.N();})

        .def("update", [](PyMonteCarlo<Scalar>& self, std::string method, int steps, int sweeps){return self.update(method, steps, sweeps);}, py::arg("method"), py::arg("steps"), py::arg("sweeps")=0)

        .def("thermalize", [](PyMonteCarlo<Scalar>& self, std::string method, int sweeps){return self.thermalize(method, sweeps);}, py::arg("method"), py::arg("sweeps"))

        .def("sample", [](const PyMonteCarlo<Scalar>& self, py::function f){return self.py_sample(f);}, py::arg("observable"))

        .def_property_readonly("data", [](const PyMonteCarlo<Scalar>& self){return self.py_data();});

    m.def("update_all",
        [](py::iterable sims, py::str method, const size_t& steps, const size_t& sweeps, const int& threads) {
            py_update_all<Scalar>(sims, method, steps, sweeps, threads);
        },
        py::arg("sims"),
        py::arg("method"),
        py::arg("steps"),
        py::arg("sweeps") = 0,
        py::arg("threads") = -1);

    //PDF STUFF
    py::class_<PyPDF<Scalar>, PyMonteCarlo<Scalar>>(m, "PDF")

        .def_property_readonly("array_data", [](const PyPDF<Scalar>& self){return self.array_data();})

        .def("draw", [](PyPDF<Scalar>& self, const size_t& samples, const size_t& therm_factor){return self.draw(samples, therm_factor);}, py::arg("samples"), py::arg("therm_factor")=0);

    py::class_<PyPDF1D<Scalar>, PyPDF<Scalar>>(m, "PDF1D")
        .def(py::init<py::function, Scalar, Scalar, Scalar>(), py::arg("pdf"), py::arg("a"), py::arg("b"), py::arg("step")=0);

    py::class_<PyPDFND<Scalar, 2>, PyPDF<Scalar>>(m, "PDF2D")
        .def(py::init<py::function, py::iterable, Scalar>(), py::arg("pdf"), py::arg("limits"), py::arg("step")=0);

    py::class_<PyPDFND<Scalar, 3>, PyPDF<Scalar>>(m, "PDF3D")
        .def(py::init<py::function, py::iterable, Scalar>(), py::arg("pdf"), py::arg("limits"), py::arg("step")=0);




    //ISING STUFF


    py::class_<SpinState<Scalar>>(m, "SpinState")
        .def_property_readonly("spins", [](const SpinState<Scalar>& self){return to_numpy<int>(self.spins, {self.shape[0], self.shape[1]});})
        .def("__call__", [](const SpinState<Scalar>& self, long int i) {
            return self(i);
        })
        .def("__call__", [](const SpinState<Scalar>& self, long int i, long int j) {
            return self(i, j);
        })
        .def_property_readonly("sites", &SpinState<Scalar>::sites)
        .def_property_readonly("M", &SpinState<Scalar>::M)
        .def_property_readonly("energy", &SpinState<Scalar>::energy);

    py::class_<PyIsingModel2D<Scalar>, PyMonteCarlo<Scalar>>(m, "IsingModel2D")
        .def(py::init<Scalar, size_t, size_t>(), py::arg("T"), py::arg("Lx"), py::arg("Ly"))
        .def_property_readonly("Temp", &PyIsingModel2D<Scalar>::T)
        .def("ssf_update", &PyIsingModel2D<Scalar>::ssf_update, py::arg("steps"), py::arg("sweeps")=0)
        .def("wolff_update", &PyIsingModel2D<Scalar>::wolff_update, py::arg("steps"), py::arg("sweeps")=0)
        .def("ssf_thermalize", &PyIsingModel2D<Scalar>::ssf_thermalize, py::arg("sweeps"))
        .def("wolff_thermalize", &PyIsingModel2D<Scalar>::wolff_thermalize, py::arg("sweeps"))
        .def("energy_sample", [](const PyIsingModel2D<Scalar>& self){return PySample(self.energy_sample());});


}



#endif




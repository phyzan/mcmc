#ifndef PDF_HPP
#define PDF_HPP



#include "mcmc.hpp"
#include "tools.hpp"
#include <array>
#include <stdexcept>
#include <string>
#include <iostream>

template<typename Scalar, size_t Dim>
using CoordsND = std::array<Scalar, Dim>; //simple alias

template<typename Scalar, size_t Dim>
using LimitsND = std::array<std::array<Scalar, 2>, Dim>; //simple alias

template<typename Scalar, size_t Dim>
Scalar choose_step(Scalar step, const LimitsND<Scalar, Dim>& limits){
    Scalar delta;

    int factor = (step == 0) ? 10 : 2;
    if (step == 0){
        step = limits[0][1] - limits[0][0];
    }
    
    if (step < 0){
        throw std::runtime_error("Step must be positive");
    }
    for (size_t i=0; i< Dim; i++){
        delta = (limits[i][1]-limits[i][0])/factor;
        if (delta <= 0){
            throw std::runtime_error("Limits invalid: " + std::to_string(limits[i][0]) + " >= " + std::to_string(limits[i][1]));
        }


        if (delta < step){
            step = delta;
        }
    }
    return step;
}


template<typename Scalar, size_t Dim>
class PDFChain final : public DerivedMarkovChain<Scalar, CoordsND<Scalar, Dim>, PDFChain<Scalar, Dim>>{

public:
    using Coords = CoordsND<Scalar, Dim>;
    using Limits = LimitsND<Scalar, Dim>;
    using Base = DerivedMarkovChain<Scalar, CoordsND<Scalar, Dim>, PDFChain<Scalar, Dim>>;
    using pdf = std::function<Scalar(const Coords&)>;

    PDFChain(const pdf& f, const Limits& limits, const Scalar& step);

    DEFAULT_RULE_OF_FOUR(PDFChain)

    void metropolis_update();

    typename Base::propagator method(const std::string& name) const {
        if (name == "metropolis"){
            return static_cast<typename Base::propagator>(&PDFChain<Scalar, Dim>::metropolis_update);
        }
        else{
            throw std::runtime_error("Method not found");
        }
    }

    const Scalar& step() const{
        return _step;
    }

    const Limits& limits() const{
        return _limits;
    }

    size_t auto_sweeps() const {
        size_t sweeps = 0;
        for (size_t i=0; i<Dim; i++){
            sweeps = std::max(size_t(10*(_limits[i][1]-_limits[i][0])/_step), sweeps);
        }
        return sweeps;
    }
    

private:

    Coords _choose_point() const;

    pdf _func;
    Limits _limits;
    Scalar _step;

};



template<typename Scalar, size_t Dim>
class MonteCarloPDF : public DerivedMonteCarlo<Scalar, CoordsND<Scalar, Dim>, PDFChain<Scalar, Dim>, MonteCarloPDF<Scalar, Dim>>{

public:

    using MC = PDFChain<Scalar, Dim>;
    using Base = DerivedMonteCarlo<Scalar, CoordsND<Scalar, Dim>, PDFChain<Scalar, Dim>, MonteCarloPDF<Scalar, Dim>>;
    using Coords = typename MC::Coords;

    MonteCarloPDF(const typename MC::pdf& f, const LimitsND<Scalar, Dim>& limits, const Scalar& step=0) : Base(PDFChain<Scalar, Dim>(f, limits, choose_step(step, limits))) {}

    DEFAULT_RULE_OF_FOUR(MonteCarloPDF)


    std::vector<CoordsND<Scalar, Dim>> draw(const size_t& samples, const size_t& therm_steps=0);
};




template<typename Scalar, size_t Dim>
PDFChain<Scalar, Dim>::PDFChain(const pdf& f, const Limits& limits, const Scalar& step) : Base(), _func(f), _limits(limits), _step(choose_step(step, limits)) {
    for (size_t i=0; i<Dim; i++){
        this->_state[i] = this->draw_uniform(this->_limits[i][0], this->_limits[i][1]);
    }
}


template<typename Scalar, size_t Dim>
CoordsND<Scalar, Dim> PDFChain<Scalar, Dim>::_choose_point() const{
    const Coords& p = this->_state;
    Coords p_new;
    Scalar a, b;
    for (size_t i=0; i<Dim; i++){
        a = _limits[i][0];
        b = _limits[i][1];
        p_new[i] = this->draw_uniform(p[i]-_step, p[i]+_step) + b-2*a;
        p_new[i] = std::fmod(p_new[i], b - a) + a;
    }
    return p_new;
}

template<typename Scalar, size_t Dim>
void PDFChain<Scalar, Dim>::metropolis_update(){
    const Coords& p = this->_state;
    Coords p_new = _choose_point();
    Scalar f_new = _func(p_new);
    Scalar f = _func(p);
    if (f_new > f || ( f_new >= this->draw_uniform(0, 1)*f)){
        this->_state = p_new;
    }
}








template<typename Scalar, size_t Dim>
std::vector<CoordsND<Scalar, Dim>> MonteCarloPDF<Scalar, Dim>::draw(const size_t& samples, const size_t& therm_steps){
    size_t sweeps = therm_steps;
    if (sweeps == 0){
        sweeps = this->_mc.auto_sweeps();
    }
    this->update("metropolis", samples, sweeps);
    return std::vector<CoordsND<Scalar, Dim>>(this->_data.end()-samples, this->_data.end());
}




#endif
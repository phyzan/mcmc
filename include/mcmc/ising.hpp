#ifndef ISING_HPP
#define ISING_HPP


#include "mcmc.hpp"

//temporarily inline
inline std::vector<int> random_spins(const size_t& Lx, const size_t& Ly){
    std::vector<int> spins(Lx*Ly);
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 1); // generates 0 or 1

    for (int& s : spins) {
        s = dist(gen) ? 1 : -1;
    }
    return spins;
}

template<typename Scalar>
struct SpinState{
    /*
    Periodic 2D spin lattice
    */
    
    std::vector<int> spins;
    std::array<size_t, 2> shape;

    SpinState(const std::vector<int>& spins, const size_t& Lx, const size_t& Ly):spins(spins), shape({Lx, Ly}){
        if (Lx*Ly != spins.size()){
            throw std::runtime_error("");
        }
    }

    inline const int& operator()(const long int& i) const;

    inline int& operator()(const long int& i);

    inline const int& operator()(long int i, long int j) const;

    inline int& operator()(long int i, long int j);

    size_t sites() const;

    Scalar M() const;

    Scalar energy() const;

    std::vector<size_t> neighbors(const size_t& site);

    size_t index(long int i, long int j) const;
};



template<typename Scalar>
class IsingModel2DMarkovChain : public DerivedMarkovChain<Scalar, SpinState<Scalar>, IsingModel2DMarkovChain<Scalar>>{

public:

    using Base = DerivedMarkovChain<Scalar, SpinState<Scalar>, IsingModel2DMarkovChain<Scalar>>;
    using propagator = typename Base::propagator;


    IsingModel2DMarkovChain(const Scalar& T, const size_t& Lx, const size_t& Ly) : Base(SpinState<Scalar>(random_spins(Lx, Ly), Lx, Ly)), _T(T), _spin_roulette(0, Lx*Ly-1){}

    DEFAULT_RULE_OF_FOUR(IsingModel2DMarkovChain)

    const Scalar& Temp() const{
        return _T;
    }

    typename Base::propagator method(const std::string& name) const;

    void ssf_update();

    void wolff_update();

private:

    Scalar _T;
    mutable std::uniform_int_distribution<size_t> _spin_roulette;

    inline size_t _choose_site() const{return _spin_roulette(this->_gen);}

};



template<typename Scalar>
class IsingModel2D : public DerivedMonteCarlo<Scalar, SpinState<Scalar>, IsingModel2DMarkovChain<Scalar>, IsingModel2D<Scalar>>{


public:

    using Base = DerivedMonteCarlo<Scalar, SpinState<Scalar>, IsingModel2DMarkovChain<Scalar>, IsingModel2D<Scalar>>;

    IsingModel2D(const Scalar& T, const size_t& Lx, const size_t& Ly): Base(IsingModel2DMarkovChain<Scalar>(T, Lx, Ly)){}

    void ssf_update(const size_t& steps, const size_t& sweeps = 0){
        this->update("ssf", steps, sweeps);
    }

    void wolff_update(const size_t& steps, const size_t& sweeps = 0){
        this->update("wolff", steps, sweeps);
    }

    void ssf_thermalize(const size_t& sweeps){
        this->thermalize("ssf", sweeps);
    }

    void wolff_thermalize(const size_t& sweeps){
        this->thermalize("wolff", sweeps);
    }

    inline Sample<Scalar> energy_sample() const{
        return this->sample([](const SpinState<Scalar>& s){return s.energy();});
    }

    inline Scalar T() const{
            return this->_mc.Temp();
    }

};
























template<typename Scalar>
inline const int& SpinState<Scalar>::operator()(long int i, long int j) const{
    return this->operator()(index(i, j));
}

template<typename Scalar>
inline int& SpinState<Scalar>::operator()(long int i, long int j) {
    return this->operator()(index(i, j));
}

template<typename Scalar>
inline int& SpinState<Scalar>::operator()(const long int& i){
    return spins[((i+spins.size()) % spins.size())];
}

template<typename Scalar>
inline const int& SpinState<Scalar>::operator()(const long int& i) const {
    return spins[((i+spins.size()) % spins.size())];
}

template<typename Scalar>
size_t SpinState<Scalar>::sites() const{
    return this->spins.size();
}

template<typename Scalar>
Scalar SpinState<Scalar>::M()const{
    return ::sum(this->spins);
}


template<typename Scalar>
Scalar SpinState<Scalar>::energy() const{
    long int res = 0;
    const SpinState& s = *this;

    for (size_t i=0; i<shape[0]; i++){
        for (size_t j=0; j<shape[1]; j++){
            res -= s(i, j)*(s(i-1, j) + s(i, j-1));
        }
    }
    return res;
}

template<typename Scalar>
std::vector<size_t> SpinState<Scalar>::neighbors(const size_t& site){
    long int i = site % shape[0];
    long int j = site / shape[0];
    return {index(i-1, j), index(i+1, j), index(i, j-1), index(i, j+1)};
}

template<typename Scalar>
size_t SpinState<Scalar>::index(long int i, long int j) const{
    i = (i + shape[0]) % shape[0];
    j = (j + shape[1]) % shape[1];
    return j * shape[0] + i;
}










template<typename Scalar>
void IsingModel2DMarkovChain<Scalar>::ssf_update(){
    SpinState<Scalar>& S = this->_state;
    size_t k = this->_choose_site();
    size_t i = k % S.shape[0];
    size_t j = k/S.shape[0];
    Scalar de = 2*S(k)*(S(i-1, j)+S(i+1, j)+S(i, j-1)+S(i, j+1));
    if (this->draw_uniform(0, 1) <= exp(-de/_T)){
        S.spins[k] *= -1;
    }
}

template<typename Scalar>
void IsingModel2DMarkovChain<Scalar>::wolff_update(){
    size_t site = _choose_site();
    SpinState<Scalar>& S = this->_state;
    const int s = S(site);
    const Scalar p = 1-std::exp(-2/_T);
    std::vector<size_t> remaining = {site}; //container with sites whose neighbors we need to check

    S(site) = -s;
    // size_t cluster_size = 1;
    while (remaining.size()>0){
        site = remaining.back(); remaining.pop_back();
        for (const size_t& nr : S.neighbors(site)){
            if ( (S(nr) == s ) && (this->draw_uniform(0, 1) < p)){
                S(nr) = -s;
                remaining.push_back(nr);
                // cluster_size++;
            }
        }
    }
}

template<typename Scalar>
typename IsingModel2DMarkovChain<Scalar>::propagator IsingModel2DMarkovChain<Scalar>::method(const std::string& name) const {
    if (name == "ssf"){
        return static_cast<propagator>(&IsingModel2DMarkovChain<Scalar>::ssf_update);
    }
    else if (name == "wolff"){
        return static_cast<propagator>(&IsingModel2DMarkovChain<Scalar>::wolff_update);
    }
    else{
        throw std::runtime_error("");
    }
}


#endif
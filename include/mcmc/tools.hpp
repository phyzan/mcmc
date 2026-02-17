#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <vector>
#include <string>
#include <cmath>
#include <omp.h>
#include <stdexcept>


#define DEFAULT_RULE_OF_FOUR(CLASSNAME)                  \
    CLASSNAME(const CLASSNAME& other) = default;      \
    CLASSNAME(CLASSNAME&& other) = default;           \
    CLASSNAME& operator=(const CLASSNAME& other) = default; \
    CLASSNAME& operator=(CLASSNAME&& other) = default;

template<typename Scalar>
std::vector<Scalar> pow(const std::vector<Scalar>& x, const Scalar& p);

template<typename Scalar>
Scalar mean_value(const std::vector<Scalar>& x);

template<typename Scalar>
Scalar sample_std(const std::vector<Scalar>& x);

template<typename Scalar>
std::vector<Scalar> bin_it(const std::vector<Scalar>& data);//assumes data is divisible by 2. Returns a vector with exactly half elements, each one is the mean of 2 consecutive elements of the original vector.

template<typename Scalar>
Scalar sum(const std::vector<Scalar>& x);

template<typename Scalar>
struct Sample;

template<typename Scalar>
struct BinningAnalysis;

template<typename Scalar>
struct Sample{

    /*
    Most important values are mean() and error():
    error() is basically the error bar of our mean value.

    For large N, the mean value follows a gaussian distribution around the true population mean,
    with a 1-sigma uncertainty of .error()

    */
    std::vector<Scalar> sample;

    Sample(const std::vector<Scalar>& sample = {}):sample(sample){}

    inline size_t N() const{ return sample.size();} //sample size

    inline Scalar mean() const{
        return mean_value(sample);
    } //sample mean value

    inline Scalar std() const{
        return sample_std(sample);
    } //standard deviation <x^2> - <x>^2

    inline Scalar popul_std() const{
        return this->std()*sqrt(this->N()/(this->N()-1.));
    } //estimate of the population std, using Bessel's correction

    inline Scalar error() const{
        return this->std()/sqrt(this->N()-1.);
    } //estimate of the gaussian 1sigma of the sample mean's distribution
    //From the CLT, this is popul_std/sqrt(N), and we can only estimate popul_std from our sample, using the formula above.

    std::string message() const;

    BinningAnalysis<Scalar> bin_it() const;


};

template<typename Scalar>
struct BinningAnalysis{

    BinningAnalysis(std::vector<Scalar> sample){_init(sample);}

    BinningAnalysis(const Sample<Scalar>& sample){_init(sample.sample);}

    size_t max_level() const{
        return _samples.size();
    }

    const bool& converged() const{
        return _converged;
    }

    Scalar tau_estimate() const{
        return 0.5 * (pow(_samples.back().error()/_samples.front().error(), 2) - 1);
    }

    const std::vector<Sample<Scalar>>& samples() const{
        return _samples;
    }

protected:

    std::vector<Sample<Scalar>> _samples;
    bool _converged;

    void _init(std::vector<Scalar> sample);

};














template<typename Scalar>
Scalar sum(const std::vector<Scalar>& x){
    Scalar res = 0;
    const Scalar* data = x.data();
    const size_t size = x.size();
    #pragma omp simd reduction(+:res)
    for (size_t i=0; i<size; i++){
        res += data[i];
    }
    return res;
}



template<typename Scalar>
std::vector<Scalar> pow(const std::vector<Scalar>& x, const Scalar& p){
    std::vector<Scalar> res(x.size());
    for (size_t i=0; i<x.size(); i++){
        res[i] = pow(x[i], p);
    }
    return res;
}

template<typename Scalar>
Scalar mean_value(const std::vector<Scalar>& x){
    return sum(x)/x.size();
}


template<typename Scalar>
Scalar sample_std(const std::vector<Scalar>& x){
    return std::sqrt(mean_value(pow(x, Scalar(2))) - pow(mean_value(x), Scalar(2)));
}

template<typename Scalar>
std::vector<Scalar> bin_it(const std::vector<Scalar>& data){
    if (data.size() % 2 != 0){
        throw std::runtime_error("Data size needs to be an even number");
    }

    std::vector<Scalar> new_data(data.size()/2);
    for (size_t i=0; i<new_data.size(); i++){
        new_data[i] = (data.at(2*i) + data.at(2*i+1))/2.;
    }
    return new_data;
}

template<typename Scalar>
std::string Sample<Scalar>::message() const{
    return std::to_string(this->mean()) + " +/- " + std::to_string(this->error());
}

template<typename Scalar>
BinningAnalysis<Scalar> Sample<Scalar>::bin_it() const{
    return this->sample;
}

template<typename Scalar>
inline Scalar abs(const Scalar& x){
    return (x >= 0) ? x : -x;
}

template<typename Scalar>
void BinningAnalysis<Scalar>::_init(std::vector<Scalar> sample){ //passing a modifiable copy
    //We cannot keep all the samples, only a power of 2
    size_t min_samples = 64;
    size_t max_level = abs(log2(sample.size()/min_samples));
    size_t max_samples = min_samples * size_t(pow(2, double(max_level)));
    //keep only the last N samples, where N is the maximum possible power of 2
    sample = std::vector<Scalar>(sample.end() - max_samples, sample.end());
    std::vector<Scalar> errors(max_level+1);
    std::vector<Scalar> rel_change(max_level);
    //Now we can bin our samples
    _samples.resize(max_level+1);
    _samples[0] = sample;
    errors[0] = _samples[0].error();
    for (size_t i=1; i<max_level+1; i++){
        _samples[i] = bin_it(_samples[i-1].sample);
        errors[i] = _samples[i].error();
        rel_change[i-1] = (errors[i]-errors[i-1])/errors[i];
    }
    if (rel_change.size() > 3){
        Scalar mean_last_change = mean_value(std::vector<Scalar>(rel_change.end()-3, rel_change.end()));
        _converged = (mean_last_change <= 0.05);
    }
    else{
        _converged = false;
    }

}

#endif
from __future__ import annotations
from typing import Iterable, Callable, TypeVar, Generic, overload
import numpy as np

class Sample:

    '''
    Sample class is a collection of data of an observable (array of numbers)
    Its methods can be used to perform statistics
    '''

    def __init__(self, data: Iterable[float]):...

    @property
    def data(self)->np.ndarray:...

    @property
    def N(self)->int:...

    @property
    def stat(self)->str:...

    def mean(self)->float:...

    def std(self)->float:...

    def popul_std(self)->float:...

    def error(self)->float:...

    def bin_it(self)->BinnedSample:...


class BinnedSample:

    '''
    When initialized, binning analysis is performed,
    and each binning level is inside .binned_samples.
    '''

    def __init__(self, data: Iterable[float]):...

    @property
    def max_level(self)->int:...

    @property
    def converged(self)->bool:...

    @property
    def tau_estimate(self)->float:...

    @property
    def binned_samples(self)->list[Sample]:...


STATE = TypeVar("STATE")


class MonteCarlo(Generic[STATE]):

    @property
    def N(self)->int:...

    @property
    def data(self)->list[STATE]:...

    def sample(self, A: Callable[[STATE], float])->Sample:...

    def update(self, method: str, steps: int, sweeps=0):...

    def thermalize(self, method: str, sweeps: int):...


class PDF(MonteCarlo[STATE]):

    @property
    def array_data(self)->np.ndarray:...

    def draw(self, samples: int, therm_factor: int=0)->np.ndarray:...


class PDF1D(PDF[float]):

    def __init__(self, pdf: Callable[[float], float], a: float, b: float, step:float=0):...


class PDF2D(PDF[tuple[float, float]]):

    def __init__(self, pdf: Callable[[float, float], float], limits: Iterable[tuple[float, float]], step:float=0):...

    def sample(self, A: Callable[[float, float], float])->Sample:...


class PDF3D(PDF[tuple[float, float, float]]):

    def __init__(self, pdf: Callable[[float, float, float], float], limits: Iterable[tuple[float, float]], step:float=0):...

    def sample(self, A: Callable[[float, float, float], float])->Sample:...



class SpinState(STATE):

    @overload
    def __call__(self, i: int)->int:...

    @overload
    def __call__(self, i: int, j: int)->int:...

    @property
    def spins(self)->np.ndarray[int]:... #2D lattice of spins +1 / -1

    @property
    def sites(self)->int:... #number of sites

    @property
    def M(self)->float:... #magnetization

    @property
    def energy(self)->float:... #total energy


class IsingModel2D(MonteCarlo[SpinState]):

    def __init__(self, T: float, Lx: int, Ly: int):...

    @property
    def Temp(self)->float:...

    def energy_sample(self)->Sample:...

    def ssf_update(self, steps: int, sweeps=0)->None:...

    def wolff_update(self, steps: int, sweeps=0)->None:...

    def ssf_thermalize(self, sweeps: int)->None:...

    def wolff_thermalize(self, sweeps: int)->None:...


#perform many Monte Carlo simulations in parallel
def update_all(sims: Iterable[MonteCarlo], method: str, steps: int, sweeps=0, threads=-1)->None:...
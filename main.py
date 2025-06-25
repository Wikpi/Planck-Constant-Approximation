import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from numpy.typing import NDArray
from pathlib import Path
import scipy.constants as cst
from scipy import odr
import scipy.signal as signal

# Measured data from the Laybold Didactic X-Ray apparatus
dataPath35: str = "data/20250624_FinalMeas1_35kV_1mA.xry"
dataPath34: str = "data/20250624_FinalMeas2_34kV_1mA.xry"
dataPath33: str = "data/20250624_FinalMeas3_33kV_1mA.xry"
dataPath32: str = "data/20250624_FinalMeas4_32kV_1mA.xry"
dataPath31: str = "data/20250624_FinalMeas5_31kV_1mA.xry"
dataPath30: str = "data/20250624_FinalMeas6_30kV_1mA.xry"
dataPath29: str = "data/20250624_FinalMeas7_29kV_1mA.xry"
dataPath28: str = "data/20250624_FinalMeas8_28kV_1mA.xry"
dataPath27: str = "data/20250624_FinalMeas9_27kV_1mA.xry"
dataPath26: str = "data/20250624_FinalMeas10_26kV_1mA.xry"

thetaUncertainty: float = 0.05 # Technically 0.1, but halved is better theoretically
potentialUncertainty: float = 50 # Technically 100, but halved is better theoretically

d: float = 283.6e-12 # 283.6 +- 3.9 pm -- specific for each target crystal
dUncertainty: float = 3.9e-12

# The inherent error in the Leybold Didactic apparatus
systematicAngleError: float = 0.1

# Custom import function
def importData(filename):
    """`importData` reads and converts .xry files to normal text data."""
    
    with open(filename, encoding="latin1") as f:
        block = f.readlines()

    lines = int(block[17][2:-1])             # number of lines in this list
    cols = int(block[17][0])                 # number of colums in this data file (if less then 10, else: [0:1])

    beta = block[4].split()                  # this row contains start and stop point of the angle beta
    beta_start = float(beta[0])
    beta_stop = float(beta[1])

    step = (beta_stop - beta_start) / (lines - 1)
    xValues = np.arange(beta_start, beta_stop + step, step)

    with open(filename, encoding="latin1") as f:
        content = f.read()

    # Use StringIO to pass the content to genfromtxt
    data = np.genfromtxt(StringIO(content), skip_header=18, max_rows=lines)

    return xValues, data

# Wavelength model from braggs law
def wavelength(angle: float) -> float:
    return 2 * d * np.sin(np.radians(angle))

# Fits planck constant approximation.
def solveUncertainty(xValues: NDArray, yValues: NDArray, error: list) -> (float, float):
    """`solveUncertainty` fits the measured data to approximate planck constant slope."""
    
    # Slope Model
    def model(B, U) -> float:
        return B[0] / U

    initial = np.array([cst.Planck * cst.speed_of_light / cst.elementary_charge])

    # ODR models
    data = odr.RealData(xValues, yValues, sx=np.array(error[0]), sy=np.array(error[1]))
    model = odr.Model(model)

    odrSetup = odr.ODR(data, model, beta0=initial)
    output = odrSetup.run()

    output.pprint()

    chi2 = output.sum_square
    print("\n Chi-squared         = ", chi2)
    chi2red = output.res_var
    print(" Reduced chi-squared = ", chi2red, "\n")

    # Final output list, first element is a list of computed values with no uncertainty: x and y, second element is their respective uncertainties.
    return [output.beta * cst.elementary_charge / cst.speed_of_light, output.sd_beta * cst.elementary_charge / cst.speed_of_light]

# Finds the start of the brehmsstrahlung continuum i.e. the highest energy lambda
def findMinLambda(angles: NDArray, intensity: NDArray) -> float:
    """`findMinLambda` finds the start of the brehmsstrahlung continuum i.e. the highest energy lambda"""
    
    # Smoothing parameters
    # n = len(intensity)
    # window_length = min(9, n if n % 2 == 1 else n - 1) # Picking an odd large enough number to smooth out the noisiness
    # polyorder = 2 if window_length > 2 else 1 # Cant be larger than the windoww length

    # # Smooth measured intensity
    # smoothed = signal.savgol_filter(intensity, window_length=window_length, polyorder=polyorder)

    # # Background value avreage that we determine to be unncecessary
    backgroundAverage: float = 4.5

    # Standard deviation of the first 10 angle intensities in the sample
    std: float = np.std(intensity[:10])
    
    mean: float = np.mean(intensity[:10])

    # Factor
    continuumFactor: float = 3

    # The min value that the algorithm should accept as the start of the continuum start
    continuumMinValue = std * continuumFactor

    # Range over all angles
    for idx, angle in enumerate(angles):
        # Reject low values
        if intensity[idx - 1] - intensity[idx - 2] < continuumMinValue:
            continue

        # Reject background same ish values
        if intensity[idx] <= backgroundAverage:
            continue

        return wavelength(angle)
    
    return 0

def findInversePotentialUncertainty(potential) -> float:
    return np.fabs(1 / (potential + potentialUncertainty) - 1 / (potential - potentialUncertainty)) / 2

# Compute wavelength specific uncertainty.
def findWavelengthUncertainty(theta) -> float:
    """`findWavelengthUncertainty` computes the uncertainty for the given angle `theta`."""
    
    dRespect = np.sin(np.radians(theta)) * dUncertainty

    thetaRespect = d * np.fabs(np.sin(np.radians(theta + thetaUncertainty)) - np.sin(np.radians(theta - thetaUncertainty)))

    uncertainty = np.sqrt(dRespect**2 + thetaRespect**2)

    return uncertainty

# Compute parameters fro planck graphing for all samples.
def computePlanckParameters(samples: list[str]) -> (NDArray, NDArray):
    """`computePlanckParameters` finds all needed parameters to successfully plot the planck constant for all samples."""
    
    # Final return values
    potentials: list[float] = []
    wavelengths: list[float] = []

    # Try all samples
    for sample in samples:
        # Get voltage from the saved filepath name i.e. data/some_data_35.xry -> voltage = 35
        filepath: str = Path(sample).stem # Take the filepath without extension
        potentialFull: str = filepath.split("_")[-2]  # Take the part after the last underscore, which is defined to be the voltage
        potential: float = float(potentialFull.replace("kV", "")) * 1000 # Remove the lettered part and ensure its in SI

        # Convert .xry data to normal text data
        xValues, yValues = importData(sample)

        # Systematic machine error
        xValues += systematicAngleError

        # Lambda min i.e. highest energy
        minLambda = findMinLambda(xValues, yValues)

        potentials.append(potential)
        wavelengths.append(minLambda)

    return np.array(potentials), np.array(wavelengths)

# Main experiment entry point.
def main() -> None:
    # All experiment samples
    samples: list[str] = [dataPath35, dataPath34, dataPath33, dataPath32, dataPath31, dataPath30, dataPath29, dataPath28, dataPath27, dataPath26]

    # Planck parameters
    potentials, wavelengths = computePlanckParameters(samples)

    potentialsError = []
    for potential in potentials:
        potentialsError.append(findInversePotentialUncertainty(potential))

    waveLengthsError = []
    for wavelength in wavelengths:
        waveLengthsError.append(findWavelengthUncertainty(wavelength))

    # Fitted planck constant
    planck, error = solveUncertainty(potentials, wavelengths, [potentialsError, waveLengthsError])

    print("planck constant after odr: ", planck)
    print("planck constant error: ", error)
    
    # We want to plot lambda vs 1 / potential
    potentials = 1.0 / potentials

    # All points of the measured data
    plt.errorbar(potentials, wavelengths, yerr=waveLengthsError, fmt="*", label='Measured', linewidth=2)

    # Actual planck constant slope
    plt.plot(potentials, (cst.Planck * cst.speed_of_light / cst.elementary_charge) * potentials, label='Expected (True h)', color='green', linestyle=':', linewidth=6)

    # Fitted planck constant slope
    plt.plot(potentials, (planck * cst.speed_of_light / cst.elementary_charge) * potentials, label='ODR Fit (λ = A / U)', color='red', linestyle='--') 

    plt.xlabel("Voltage 1/U (V)")
    plt.ylabel("Wavelength λ (m)")
    plt.title("Fitting Planck's Constant (%d samples)" % len(samples))
    
    plt.grid(True)
    plt.legend()

    plt.savefig("saved/%s" % "Fitting Planck's Constant (%d samples)" % len(samples))

    plt.show()

main()

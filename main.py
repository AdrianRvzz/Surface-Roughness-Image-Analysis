import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import math
from scipy import integrate


def FillArrayWithPixelCoords(array, color, pathImageFile):
    image = cv2.imread(pathImageFile, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    array = np.argwhere(thresh == color)

    return array


def ShowImage(pathImageFile):
    image = cv2.imread(pathImageFile)
    filename = pathImageFile[pathImageFile.rfind('/') + 1:]
    cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
    cv2.imshow(filename, image)


def SplitInXYArray(array):
    y, x = zip(*array)
    y = np.array(y)
    x = np.array(x)
    #translate to x=0
    x=x-np.min(x)
    y=y-y[0]+x

    return x, y * -1


def ObtainPolyOfRegression(x, y, grade):
    coEffs = np.polyfit(x, y, grade)
    poly = np.poly1d(coEffs)
    return poly


def SizeOfPoint(x):
    return 100 * (1 / len(x))


def PointYToOrigin(x, y, poly):
    # Ajustar una regresión lineal a los datos
    x_centered = x  # - x.mean()
    y_centered = y - poly(x)  # + poly(x.mean())
    return x_centered, y_centered


whitepixels = []
white = 255
pathImageFile = "images/44/44.1/CircleSection.tif"

PixelsCoords = FillArrayWithPixelCoords(whitepixels, white, pathImageFile)
PixelInX, PixelInY = SplitInXYArray(PixelsCoords)

print("X: ", PixelInX, "Y: ", PixelInY)

# Poly of regression
poly = ObtainPolyOfRegression(PixelInX, PixelInY, 3)
PointXToOrigin, PointYToOrigin = PointYToOrigin(PixelInX, PixelInY, poly)


def SetEscale(x, y, unit):
    return x * unit, y * unit


unit = 2.06
PointXToOrigin, PointYToOrigin = SetEscale(PointXToOrigin, PointYToOrigin, unit)


def GetWaviness(y, alpha, lambdaCutoff):
    sigma = lambdaCutoff / (2 * np.log(1 / alpha))
    w = gaussian_filter1d(y, sigma=sigma, mode='nearest')
    return w


def GetRoughness(w, p):
    return p - w

def GetWaviness2(x, y, alpha, lambdaCutoff, profileFunction):

    y_weight = np.zeros_like(y)

    sampling_length = np.max(x)
    print("Cutoff: ", lambdaCutoff, "microns")
    print("Lenght: ", sampling_length, "microns")
    lenght_in_cutOffs = sampling_length / lambdaCutoff
    print("Lenght in cutoffs: ", lenght_in_cutOffs)
    lenght_in_cutOffs -= 1
    print("Final lenght in cutoffs: ", lenght_in_cutOffs)
    minInterval = lambdaCutoff / 2
    print("Min interval: ", "0-", minInterval)
    maxInterval = sampling_length - lambdaCutoff / 2
    print("Max interval: ", maxInterval, "-", sampling_length)

    # 0.8 = 0.0008 microns
    # Mayor cutoff, mas plano el waviness
    # Menor cutoff, mas detallado el waviness
    # En teoria

    lambdaCutoff /= 1000

    # print(x_filtered)
    for i in range(len(x)):
        # print(profileFunction(x[i]))
        # print(x_filtered[i])
        # print(profileFunction([i]))
        # print(profileFunction(x_filtered[i]))
        # w[i] = (1 / (alpha * lambdaCutoff)) * math.exp(-math.pi * ((y_nuevo[i]/ (alpha * lambdaCutoff)) ** 2))
        # w[i]= ()*(math.exp(-math.pi))
        # w[i] = math.exp(-math.pi * (y_nuevo[i] / (alpha * lambdaCutoff)) ** 2) / (alpha * lambdaCutoff)
        # y_weight[i] = math.exp(-math.pi * (profileFunction(x[i]) / (alpha * lambdaCutoff)) ** 2) / (alpha * lambdaCutoff)
        # if(x[i]>=minInterval and x[i]<maxInterval):

        #exponencial_value = math.exp(-1 * math.pi * (pow(2, profileFunction(x[i]) / (alpha * lambdaCutoff))))
        exponencial_value = np.exp(-1 * np.pi * (np.power(2, profileFunction(x[i]) / (alpha * lambdaCutoff))))

        # print(x[i])
        # exponencial_value = math.exp(-1 * math.pi * (pow(2, profileFunction(x[i])/ (alpha * lambdaCutoff))))
        # x_filtered.append(x[i])
        # print(exponencial_value)
        y_weight[i] = (1 / alpha * lambdaCutoff) * exponencial_value

        # y_weight[i]=(1/alpha*lambdaCutoff)*np.exp(-np.pi*(profileFunction(x[i])/alpha*lambdaCutoff)**2)
        # print(y_weight[i])

    # print(len(x_filtered))

    weight_function = interp1d(x, y_weight, kind="zero")
    # print("X: ", x, "y: ", y_weight)
    # f_convolve =np.convolve(profileFunction(x), weight_function(x), mode='same')
    f_convolve = np.convolve(profileFunction(x), weight_function(x), mode='same')
    # f_convolve = convolve(profileFunction(x), y_weight, mode='same')

    # print("F convolve values: ", f_convolve)
    f_waviness = interp1d(x, f_convolve, kind='quadratic')
    return f_waviness


# Función para calcular la línea media


alpha = 0.4697
lambdaCutoff = 25


def SortAndUniqueCoords(x, y):
    x_unique = np.unique(x)
    x_unique = x_unique -  min(x_unique)

    sorted_indices = np.argsort(x_unique)
    x_unique_sorted = x_unique[sorted_indices]
    # calcular el promedio de las coordenadas Y para cada valor único de X
    y_mean = [np.mean([y[j] for j in np.where(x == i)[0]]) for i in x_unique_sorted]

    return x_unique_sorted, y_mean


def InterpolateFunction(x, y):
    x_unique_sorted, _ = SortAndUniqueCoords(x, y)
    # calcular el promedio de las coordenadas Y para cada valor único de X
    _, y_mean = SortAndUniqueCoords(x, y)

    return interp1d(x_unique_sorted, y_mean, kind="quadratic")


x_unique, _ = SortAndUniqueCoords(PointXToOrigin, PointYToOrigin)
_, y_mean = SortAndUniqueCoords(PointXToOrigin, PointYToOrigin)

profile_function = InterpolateFunction(x_unique, y_mean)
waviness_function = GetWaviness2(x_unique, y_mean, alpha, lambdaCutoff, profile_function)

RoughnessY = PointYToOrigin
#ShowImage(pathImageFile)

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

x = x_unique
y = y_mean


def update_plot():
    # Obtener los valores de los cuadros de texto
    try:
        lambda_cutoff = float(lambda_entry.get())
        alpha = float(alpha_entry.get())
    except ValueError:
        return
    #f_waviness = GetWaviness2(x, y, alpha, lambda_cutoff, profile_function)
    #f_gaussian = gaussian_filter1d(y, sigma=lambda_cutoff, mode='nearest')
    for ax in axs:
        ax.clear()
        ax.set_ylim(min(y), max(y))
##
    f_gaussian = gaussian_filter1d(y, sigma=lambda_cutoff, mode='nearest')
    f_roughness = GetRoughness(f_gaussian, y)

    func_gaussian = interp1d(x_unique, f_gaussian, kind='cubic')
    func_roughness = interp1d(x_unique, f_roughness, kind='cubic')
    func_profile = interp1d(x_unique, y, kind="cubic")
    # area=integrate(f_profile(x))

    print("funciones: ", func_gaussian, func_roughness, func_profile)

    data = {"Form": (PixelInX, PixelInY, "black"),
            "Profile": (x_unique, func_profile, "blue"),
            "Waviness": (x_unique, func_gaussian ,"orange"),
            "Roughness": (x_unique, func_roughness, "red")}

    ##

    sigma=lambda_cutoff
    #f_gaussian = gaussian_filter1d(y, sigma=sigma, mode='nearest')
    #f_roughness = GetRoughness(f_gaussian, y)

   # data = {"Form": (PixelInX, PixelInY, "black"),
    #        "Profile": (x_unique, y, "blue"),
     #       "Waviness": (x_unique, f_gaussian, "orange"),
     #       "Roughness": (x_unique, f_roughness, "red")}

    # Iterar sobre los subplots y los datos correspondientes

    for ax, (title, (x, ydata, color)) in zip(axs, data.items()):
        if title == "Form":
            ax.scatter(PixelInX * unit, PixelInY * unit, label='Form', s=SizeOfPoint(PixelInX))
            ax.set_yticklabels([])
            axs[0].axhline(y=0, color="black")
            # axs[0].scattr(x, poly(x), label="Form")
            ax.set_title("Form")
            ax.legend(markerscale=30)
            ax.set_ylim(min(PixelInY * unit), max(PixelInY * unit))
        else:
            ax.plot(x, ydata(x), label=title, color=color)  # Plotear los datos
            ax.axhline(y=0, color="black")  # Agregar una línea horizontal en y=0

            if(alpha>max(x)):
                alpha=max(x)
            intervals= np.max(x)/alpha

            print("Intervals: ", intervals)
            intervals = math.floor(intervals)
            print("Intervals: ", intervals)

            #if(intervals<5):
            distanceInterval = max(x)/intervals


            print("---------------------------\n")
            print("\n", title)
            max_peaks = np.zeros(intervals)
            max_valleys= np.zeros(intervals)
            z_sum = np.zeros(intervals)

            x_peaks = np.zeros(intervals)
            x_valleys= np.zeros(intervals)

            for i in range(intervals):

                inicio = i* distanceInterval
                fin = (i + 1) * distanceInterval
                ax.axvline(x=inicio, color='r')
                ax.axvline(x=fin, color='r')
                print(inicio, "-", fin)

                x_eval=np.linspace(inicio, fin, int(alpha))

                max_peaks[i]=np.max(ydata(x_eval))
                max_valleys[i]=np.min(ydata(x_eval))

                z_sum[i] = abs(np.max(ydata(x_eval))) + abs(np.min(ydata(x_eval)))
                x_peaks [i]= x_eval[np.argmax(ydata(x_eval))]
                x_valleys [i]= x_eval[np.argmin(ydata(x_eval))]


            ax.scatter(x_peaks, max_peaks)
            ax.scatter(x_valleys, max_valleys)
            print("Valleys : ", max_valleys)
            print("Peaks: ", max_peaks)
            print("Z sum: ", z_sum)
            print("-----------------\n")

            ax.set_title(title)  # Establecer el título del subplot
            ax.axhline(y=average_height(ydata(x)), color="red", linestyle="--",
                       label=title[0] + "a = {:.3f}".format(
                           average_height(ydata(x))))  # Agregar la línea de la media aritmética
            ax.axhline(y=quadratic_average(x, ydata(x)), color="green", linestyle="--",
                       label=title[0] + "q = {:.3f}".format(
                           quadratic_average(x, ydata(x))))  # Agregar la línea de la media cuadrática

            minValley, maxValley = PeakAndValleyHigher(ydata(x))
            #zOnly = abs(average_height(max_valleys))+abs(average_height(max_peaks))

            ax.axhline(y=minValley, color="red", linestyle="-",
                       label=title + "max valley = {:.3f}".format(minValley), linewidth=2)

            ax.axhline(y=maxValley, color="green", linestyle="-",
                       label=title + "max peak = {:.3f}".format(maxValley), linewidth=2)

            ax.axhline(y=DistanceInPeakAndValley(minValley, maxValley), color="green", linestyle="-",
                       label=title + "z valle and peak = {:.3f}".format(DistanceInPeakAndValley(minValley, maxValley)),
                       linewidth=2)

            ax.axhline(y= average_per_lenght(z_sum), linestyle="-",
                       label=title + "z = {:.3f}".format(average_height(z_sum)), linewidth=0.5)

            ax.axhline(y= average_per_lenght(max_valleys),  linestyle="-",
                       label=title + "v = {:.3f}".format(average_height(max_valleys)), linewidth=0.5)
            ax.axhline(y= average_per_lenght(max_peaks), linestyle="-",

                       label=title + "p = {:.3f}".format(average_height(max_peaks)), linewidth=0.5)
            ax.legend()  # Agregar la leyenda

    canvas.draw()

def average_height( y):
    # Calculamos la magnitud de la desviación absoluta

    abs_deviation = np.abs(y)
    # Calculamos Ra
    #L = np.max(x) - np.min(x)
    ave = np.mean(abs_deviation)
    return ave

def average_per_lenght(y):
    if len(y)==1:
        return y[0]
    return np.mean(y)
def quadratic_average(x, y):
    L = np.max(x) - np.min(x)
    # Calculaos el valor RMS ----------------
    cuadratic_deviation = np.square(y)
    qms = np.sqrt(abs((1 / L) * integrate.simps(cuadratic_deviation, x)))
    return qms

def PeakAndValleyHigher( y):
    minimum=np.min(y)
    maximum=np.max(y)
    return minimum, maximum


def DistanceInPeakAndValley(min, max):
    return abs(min)+abs(max)



root = tk.Tk()
root.title('Form, Profile, Waviness and Roughness')



frame1 = tk.Frame(root)
frame1.pack(side=tk.TOP)

lambda_label = tk.Label(frame1, text='Sigma:')
lambda_label.pack(side=tk.LEFT)

lambda_entry = tk.Entry(frame1)
lambda_entry.insert(0, str(lambdaCutoff))
lambda_entry.pack(side=tk.LEFT)


alpha_label = tk.Label(frame1, text='LambdaCutoff(mm):')
alpha_label.pack(side=tk.LEFT)

alpha_entry = tk.Entry(frame1)
alpha_entry.insert(0, str(lambdaCutoff))
alpha_entry.pack(side=tk.LEFT)


button = tk.Button(frame1, text='Plot', command=update_plot)
button.pack(side=tk.LEFT)

frame2 = tk.Frame(root)
frame2.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


fig, axs = plt.subplots(4, 1, sharex=True, figsize=(5, 8))


canvas = FigureCanvasTkAgg(fig, master=frame2)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Crear la barra de herramientas
toolbar = NavigationToolbar2Tk(canvas, frame2)
toolbar.update()

fig.text(0.5, 0.04, 'µm', ha='center', va='center')
fig.text(0.06, 0.5, 'µm', ha='center', va='center', rotation='vertical')

f_gaussian = gaussian_filter1d(y, sigma=20, mode='nearest')
f_roughness = GetRoughness(f_gaussian, y)

f_profile = interp1d(x, y, kind='quadratic')

func_gaussian = interp1d(x, f_gaussian, kind='quadratic')
func_roughness = interp1d(x, f_roughness, kind='quadratic')


#area=integrate(f_profile(x))

for ax in axs:
    ax.set_ylim(min(y), max(y))


data = {"Form": (PixelInX ,PixelInY, "black"),
        "Profile": (x, f_profile(x), "blue"),
        "Waviness": (x, func_gaussian(x), "orange"),
        "Roughness": (x, func_roughness(x), "red")}




# Iterar sobre los subplots y los datos correspondientes
for ax, (title, (x, ydata, color)) in zip(axs, data.items()):
    if title=="Form":
        ax.scatter(PixelInX*unit, PixelInY*unit, label='Form', s=SizeOfPoint(PixelInX))
        ax.set_yticklabels([])
        axs[0].axhline(y=0, color="black")
        # axs[0].scattr(x, poly(x), label="Form")
        ax.set_title("Form")
        ax.set_ylim(min(PixelInY*unit), max(PixelInY*unit))
        ax.legend(markerscale=30)

    else:
        ax.plot(x, ydata, label=title, color=color, linestyle="dashdot")  # Plotear los datos
        ax.axhline(y=0, color="black")  # Agregar una línea horizontal en y=0
        #ax.axvline(x=500, color="black")
        ax.set_title(title)  # Establecer el título del subplot
        ax.axhline(y=average_height(ydata), color="red", linestyle="--",
               label=title[0]+"a = {:.3f}".format(average_height(ydata)))  # Agregar la línea de la media aritmética
        ax.axhline(y=quadratic_average(x, ydata), color="green", linestyle="--",
               label=title[0]+"q = {:.3f}".format(quadratic_average(x, ydata)))  # Agregar la línea de la media cuadrática

        minValley=PeakAndValleyHigher(ydata)[0]
        maxValley=PeakAndValleyHigher(ydata)[1]

        ax.axhline(y=minValley, color="red", linestyle="-",
                   label=title[0] + "max valley = {:.3f}".format(PeakAndValleyHigher(ydata)[0]),  linewidth=2)

        ax.axhline(y=maxValley, color="green", linestyle="-",
                   label=title[0] + "max peak = {:.3f}".format(PeakAndValleyHigher(ydata)[1]), linewidth=2)

        ax.axhline(y=DistanceInPeakAndValley(minValley, maxValley), color="purple", linestyle="-",
                   label=title[0] + "z = {:.3f}".format(DistanceInPeakAndValley(minValley, maxValley)), linewidth=0)

        ax.legend()
        #ax.legend(labels=title[0] + "z = " + str(DistanceInPeakAndValley(minValley, maxValley)))
        #ax.legend(label=title[0]+"z = "+ DistanceInPeakAndValley(minValley, maxValley))   # Agregar la leyenda


# Iniciar la aplicación
root.mainloop()

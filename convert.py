import scipy.io
import SimpleITK as sitk

for i in range(1,21):
    number = str(i)
    if i < 10:
        number = "0" + number
    path = r"vessel12dataset\imgs\VESSEL12_" + number
    pathwexst = path + ".mhd"
    img = sitk.GetArrayFromImage(sitk.ReadImage(pathwexst))

    scipy.io.savemat(path + ".mat", {"img" : img})


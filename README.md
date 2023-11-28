# PET-CT_ACR_phantom_analysis
Automated SUV analysis of the PET-CT ACR phantom.

Environment: **Windows 10, Python 3.7.9**

In the command line, run the following:
```console
foo@bar:~$ python3 -m venv env
foo@bar:~$ .\env\Scripts\activate
foo@bar:~$ pip install -r requirements.txt
foo@bar:~$ python PET-CT_analysis.py
```

Or open up the [Jupyter Notebook](PET-CT_analysis.ipynb) to look at it by sections.

There are several notes hidden within the scipt file itself, be sure to check these out if you run into any errors.

Some basic notes:
1. I used CT and PET data which was isotropic and 133 slices each, I think it will also work for non-isotropic and unequal slicing due to its reliance on the DICOM parameter `ImagePositionPatient` which converts everything to a shared coordinate system.
2. My algorithm for finding the circles works great for my sample images (which seem like good representatives of PET/CT data for ACR phantom analysis), however they may not work as well for other acquisitions and the parameters may need to be adjusted, especially the non-radius parameters (i.e., `C_PARAM`) which extract the contrast differences.
3. My passing criteria is based on the 2010 update to the ACR guidelines, however if new updates arise, then the `PASSING_CRITERIA` global variable will have to be adjusted accordingly.
4. Double check the global variables first such as `CT_SLICE_LOCATION` which defines the slice location of the vials in the CT image ... I wish to make this automatic in the future. `REFERENCE_VIAL_SIZES` should not change unless the ACR decides to update their phantom. `PATIENT_WEIGHT` (70 kg) and `DELIVERED_DOSE` (10 mCi) are based on the ACR guidelines as of 2023.
5. Inside the `data/CT` and `data/PET` folders should be the DICOM images for each slice (i.e., 133 DICOM images for my case) without any parent folders so `data/CT/im1.dcm`, `data/CT/im2.dcm`, etc.

## Data
Here's where I am storing small dedicated datasets for the ChemometricsTools package. To view a list of datasets that are available in this package with-in the Julia environment you can run the following command:
```Julia
ChemometricsToolsDatasets()
```

To conveniently load a dataset you can enter either the number for the respective dataset or the filename:
```Julia
ChemometricsToolsDataset("tecator.csv")
```

You will notice that for datasets that require permissions for redistribution (Ex: Tecator), everytime it is loaded a message is displayed conveying this fact. I am an appreciator of open source, but data collection takes a lot of time, and I want to respect those efforts by reminding end-users of copyrights, ownership, giving appropriate credit, etc.

## Your data could be here!
Please reach out to me if you have a chemical dataset that you would like to share for purposes of this package.

## Is your data here and you don't want it to be?
Please file an issue on github and I will resolve it as promptly as possible.

## Permissions and Credit:

#### Pharma_Instrument1.csv and Pharma_Instrument2.csv
These files contain NIR spectra of 653 pharmaceutical tablets from the 2002 International Diffuse Reflectance Conference. The data was collected with 2 spectrometers with a spectral range of 600-1398nm with a resolution of 2nm. Permission was granted to the author on LinkedIn (several months ago), where it was stated the data was free to redistribute. 

#### tecator.csv
*Statement of permission from Tecator (the original data source).These data are recorded on a Tecator Infratec Food and Feed Analyzer working in the wavelength range 850 - 1050 nm by the Near Infrared Transmission (NIT) principle. Each sample contains finely chopped pure meat with different moisture, fat and protein contents.If results from these data are used in a publication we want you to mention the instrument and company name (Tecator) in the publication. In addition, please send a preprint of your article to Karin Thente, Tecator AB, Box 70, S-263 21 Hoganas, Sweden. The data are available in the public domain with no responsability from the original data source. The data can be redistributed as long as this permission note is attached. For more information about the instrument - call Perstorp Analytical's representative in your area.*

#### actuator.csv
Borrowed from : https://c3.nasa.gov/dashlink/resources/503/
Contributing Author(s) : Edward Balaban , Sriram Narasimhan , Indranil Roychoudhury , Abhinav Saxena
A data set of two-minute scenarios with a large set of motion and load profiles. Each scenario started with a nominal actuator, then, at 1-minute mark the ballscrew jam fault was injected by switching to the faulty actuator. The scenarios are intended for testing diagnostic algorithms. Please note that only the low speed data (current, temperature, position, and load) were collected during this set of experiments.

#### iris.csv
Fisher,R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188
(1936); also in "Contributions to Mathematical Statistics" (John Wiley, NY, 1950).

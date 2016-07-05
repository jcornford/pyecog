import glob, os, numpy
import stfio

def main():

    
    
    searchpath = os.getcwd()
    exportdirectory = searchpath+'/ConvertedFiles/'

    # Make export directory
    if not os.path.exists(exportdirectory):
        os.makedirs(exportdirectory)
        
    # Walk through and find abf files   

    pattern = '*.abf'
    datafilenames = glob.glob(pattern)
    if datafilenames:
        for filename in datafilenames:
            print 'Converting ',filename
            data = stfio.read(filename,ftype = "abf")
            x = data.aspandas()
            x = x.values
            numpy.save(exportdirectory+filename[0:-4],x)

           

if __name__ == '__main__':
  main()
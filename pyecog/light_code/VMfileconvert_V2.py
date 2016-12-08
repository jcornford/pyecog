import glob, os, numpy, sys
try:
    import stfio
except:
    sys.path.append('C:\Python27\Lib\site-packages')
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
            print ('Converting '+str(filename))
            data = stfio.read(filename,ftype = "abf")
            x = data.aspandas()
            x = x.values
            numpy.save(exportdirectory+filename[0:-4],x)

           

if __name__ == '__main__':
  main()

import sys
import os

#---------------------------------------------------------------------------------------------#
#
# Writes the Nozzle properties in the text file <Nozzle_prop.out>
#

# def WriteDataFile(Operation,VAR1,File_Name,flag='Nrm'):

#     f=open(File_Name,Operation)
#     if Operation == 'w':
#         f.write('x\ty\tu\tv\tV\tM\trho\ta\tT\tP\tQ\n')
#     elif Operation == 'a':
#         if flag == 'Nrm':
#             for i in VAR1:
#                 for j in i:
#                     f.write("%f %f %f %f %f %f %f %f %f %f %f\n" %(j.x,j.y,j.u,j.v,j.V,j.M,j.rho,j.a,j.Temp,j.Press,j.Quality))
#         elif flag == 'TAB':
#             f.write("%d %f %f %f %f %f %f %f\n" %(VAR1.V,VAR1.M,VAR1.rho,VAR1.a,VAR1.Temp,VAR1.Press,VAR1.gamma))
#         else:
#             print('*** Unknow Flag ***')
#             sys.exit(-1)
#     f.close()

# #---------------------------------------------------------------------------------------------#
# #
# # Writes the Nozzle co-ordinates in the text file <Nozzle_coords.out>
# #

# def WriteNozzleDim(Coords,File_Name):
#     try:os.remove(os.getcwd()+'/'+File_Name)
#     except: print ("Continue: No File to Delete <Co-ordinate file>\n Location:",os.getcwd())
#     f=open(File_Name,'w')
#     for i in Coords:
#         f.write("%f %f %f\n" %(i.x*1000,i.y*1000,0.0))
#     f.close()

#---------------------------------------------------------------------------------------------#
#
# Prints the progress bar on the screen [shows the percentage calcualtion left]
#

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100, on_progress = None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        on_progress - Optional  : callable(fraction: float, prefix: str) -> None,
                                   called with the same (clamped 0..1) fraction
                                   used for the terminal bar. Lets a caller
                                   (e.g. a Dash background callback's
                                   set_progress) mirror this same progress
                                   stream to a UI progress bar without
                                   touching the terminal output at all --
                                   see moc.progress.
    """
    fraction        = min(1.0, iteration / float(total))
    filledLength    = int(round(barLength * fraction))
    percents        = round(100.00 * fraction, decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        print("\n")
    if on_progress is not None:
        on_progress(fraction, prefix)

##
## END
##---------------------------------------------------------------------------------------------#


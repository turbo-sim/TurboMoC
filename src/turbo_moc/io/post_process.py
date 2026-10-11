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
    """Deprecated: forward loop progress to `on_progress` only.

    The former carriage-return terminal bar is gone (it concatenated in
    captured output); the solvers now report through turbo_moc.reporting.
    Kept so external callers keep working: ``on_progress(fraction, prefix)``
    receives the clamped fraction ``iteration / total`` (1 when total <= 0).
    The other arguments are accepted and ignored.
    """
    fraction = 1.0 if total <= 0 else min(1.0, max(0.0, iteration / float(total)))
    if on_progress is not None:
        on_progress(fraction, prefix)

##
## END
##---------------------------------------------------------------------------------------------#


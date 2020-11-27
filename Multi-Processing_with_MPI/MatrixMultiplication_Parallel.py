# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from mpi4py import MPI
import numpy as np
import math

comm=MPI.COMM_WORLD

"""build random arrays"""
Size_Of_Array=10000000
rank=comm.Get_rank()
"""Get number of processes"""
Num_Of_Processes=comm.Get_size()
"""making sure that the size is >= 2^d"""

if (Num_Of_Processes)%2==0:	
    """calculate size of each small array"""
    #print('num of proc.',Num_Of_Processes  )
    if(rank==0):
     Start_Time=MPI.Wtime()    
     if (2*rank)+1 < Num_Of_Processes:
         sent_Array=np.random.randint(1,10,Size_Of_Array)
         comm.send(sent_Array,(2*rank)+1)
        	
     if (2*rank)+2 < Num_Of_Processes:
         sent_Array=np.random.randint(1,10,Size_Of_Array)
         comm.send(sent_Array,(2*rank)+2)  
     print('heree')
     comm.Barrier()	
     End_Time=MPI.Wtime()
     print('Time collapsed=',End_Time-Start_Time)
              
    if rank!=0:
     array=comm.recv(source=int(rank-1)/2)
     #print('recv array')
     if (2*rank)+1 < Num_Of_Processes:
         sent_Array=np.random.randint(1,10,Size_Of_Array)
         comm.send(sent_Array,(2*rank)+1)
        	
     if (2*rank)+2 < Num_Of_Processes:
         sent_Array=np.random.randint(1,10,Size_Of_Array)
         comm.send(sent_Array,(2*rank)+2)	
        
     comm.Barrier()	
	
elif (Num_Of_Processes)%2!=0:
	print('This size cannot be handeled in recursive way!!')

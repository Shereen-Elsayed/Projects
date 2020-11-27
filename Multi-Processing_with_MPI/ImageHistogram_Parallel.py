from mpi4py import MPI
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

comm=MPI.COMM_WORLD


rank=comm.Get_rank()
"""Get number of processes"""
Num_Of_Processes=comm.Get_size()
"""calculate size of each small array"""
#print('num of proc.',Num_Of_Processes  )
Matrix=None
Sub_Matrix=None
Size_Of_Sent_Rows=None
To_Send=list()


if rank ==0:
        Start_Time=MPI.Wtime()
        Image = np.asarray(Image.open("/media/Shereen/HieldshiemMasters/Semester1/DistributedDataAnalytics/Exercises/Ex2_Solution/Ex2_ShereenElsayed/Image.jpg"))
        imgdim=np.shape(Image)
        print('imgdim:   ',imgdim)
        """build random arrays"""
        N=imgdim[0]
        #image is gray level
        if len(imgdim) ==2: 
              Matrix=Image.reshape(imgdim[0],imgdim[1],1)
        else:
              Matrix=Image
        Matrixdim=np.shape(Matrix)
        print('Matrix dim=',np.shape(Matrix))            
        if (N%Num_Of_Processes) == 0:
            Size_Of_Sent_Rows=N/Num_Of_Processes
            mod=0
        elif (N%Num_Of_Processes) != 0:
             mod=N%Num_Of_Processes
             #Add dummy rows so it will be divisable
             while N%Num_Of_Processes !=0:
                   temp=np.zeros((1,Matrixdim[1],Matrixdim[2]), dtype=int)
                   temp=temp-1
                   Matrix=np.concatenate([Matrix,temp])
                   N=N+1
             Size_Of_Sent_Rows=N/Num_Of_Processes
             #print('Size_Of_Sent_Rows  ',Size_Of_Sent_Rows)
             #print('mooood', mod)
        
        start=0
        end=Size_Of_Sent_Rows
        for i in range(0,Num_Of_Processes):
              var=Matrix[int(start):int(end),:,:]
              To_Send.append(var)
              print('----------',len(To_Send))
              start=start+Size_Of_Sent_Rows
              end=end+Size_Of_Sent_Rows

Sub_Matrix=comm.scatter(To_Send,root=0)
#print('rank  ======',rank,'sub mat======',Sub_Matrix)
#Sub_Matrix=comm.gather(Sub_Matrix,root=0)

dim=np.shape(Sub_Matrix)
#print('rank  ======',rank,'sub mat======',dim)
Histo_Count=np.zeros((256,dim[2]),dtype=int) 
for i in range(0,dim[0]):
    for j in range(0,dim[1]):
         for y in range(0,dim[2]):
               if Sub_Matrix[i][j][y]!=-1:
                   Histo_Count[Sub_Matrix[i][j][y],y]=Histo_Count[Sub_Matrix[i][j][y],y]+1

#print('rank  ======',rank,'sub mat======',Histo_Count)
#comm.send(Histo_Count,0)  

Result=comm.gather(Histo_Count,root=0)
print('Result',np.shape(Result))


if rank==0:        
         #calculation part
        #Final_Histo=np.zeros((256,Matrixdim[2]),dtype=int)      
        
        Final_Histo=np.sum(Result,axis=0)
        #print('Final Result',np.shape(Final_Histo))       
        End_Time=MPI.Wtime()
        #lines=plt.plot(Final_Histo[:,[0]])
        #plt.setp(lines, color='r', linewidth=2.0) 
        #plt.show()   
        print('Time Collapsed',End_Time-Start_Time)




    


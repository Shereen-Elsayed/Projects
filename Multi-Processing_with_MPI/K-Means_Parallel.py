from mpi4py import MPI
import numpy as np
import math
import csv


comm=MPI.COMM_WORLD

"""build random arrays"""
No_Of_Clusters=8
Number_Of_Iterations=4
rank=comm.Get_rank()
"""Get number of processes"""
Num_Of_Processes=comm.Get_size()
Cluster_centroids=None
No_Of_Features=None
Start_Time=0.0
if rank ==0:
    #need to read it from the data set
    Training_Data=np.loadtxt(open("/media/Shereen/HieldshiemMasters/Semester1/DistributedDataAnalytics/Exercises/Ex3_Solution/Absenteeism_at_work_AAA/Absenteeism_at_work.csv", "rb"), delimiter=";", skiprows=1)
    #print('Training',Training_Data)   
    Start_Time=MPI.Wtime()
    #print('Training_Data',Training_Data)
    dim=np.shape(Training_Data)
    No_Of_Features=dim[1]
    No_Of_Samples=dim[0]
    Cluster_centroids=np.random.randint(1,50,[No_Of_Clusters,No_Of_Features])
    if (No_Of_Samples%Num_Of_Processes) == 0:
        Size_Of_Sent_Samples=No_Of_Samples/Num_Of_Processes
        mod=0
    elif (No_Of_Samples%Num_Of_Processes) != 0:
            mod=No_Of_Samples%Num_Of_Processes
            Size_Of_Sent_Samples=math.floor(No_Of_Samples/Num_Of_Processes)
    start=0
    end=Size_Of_Sent_Samples
    sub_samples=Training_Data[int(start):int(end)]
    start=start+Size_Of_Sent_Samples
    end=end+Size_Of_Sent_Samples
    for i in range(1,Num_Of_Processes):
        sent_Samples=Training_Data[int(start):int(end)]
        if i==(Num_Of_Processes-1):
            sent_Samples=Training_Data[int(start):int(No_Of_Samples)]
            
        #print('sent array ',sent_Rows)
        comm.send(sent_Samples,i)
        start=start+Size_Of_Sent_Samples
        end=end+Size_Of_Sent_Samples

Cluster_centroids=comm.bcast(Cluster_centroids,root=0)


for w in range (Number_Of_Iterations):
        Membership=[list()for i in range(No_Of_Clusters)]
        Local_mean=[list()for i in range(No_Of_Clusters)]
        if rank !=0:
             if w==0:
                  sub_samples= comm.recv(source=0)
	     #print('samples=======',np.shape(sub_samples))
	#########################Calculate the Distance matrix###############

        Data_Size=np.shape(sub_samples)
        Centroids_Size=np.shape(Cluster_centroids)
	#print('sub_samples',sub_samples,'rank=',rank)
        distance=0
        min_distance=9999
        min_index=None
        for i in range(0,Data_Size[0]):
                for j in range(0,Centroids_Size[0]):
                    for z in range(0,Data_Size[1]):
                        distance=distance+(sub_samples[i][z]-Cluster_centroids[j][z])**2
                    distance=math.sqrt(distance)
                    #print('distanceeee',distance)
                    #print('min distanceeee',min_distance)
                    if distance<min_distance:
                       min_index=j
                       min_distance=distance 
                #print('min index',min_index)          
                Membership[min_index].append(sub_samples[i])
                #print('min_index',min_index,'rank=',rank)
                min_distance=9999
        #if w==Number_Of_Iterations-1:
               #print('Membership for cluster 0',Membership[1],'rank=',rank)
        mean=0
        sub_mean=[0]*Data_Size[1]
        memdim=np.shape(Membership)
        #print('----------',np.shape(memdim))
	##############Calculate the local means##############################
        for i in range(0,len(Membership)):
            if  len(Membership[i])>0: 
                 for z in range(0,Data_Size[1]):
                      for j in range(0,len(Membership[i])):
                          mean=mean+Membership[i][j][z]
                      sub_mean[z]=mean/len(Membership[i])
                 Local_mean[i].append(sub_mean)	    	            																	
	#print('Local_mean',Local_mean,'rank=',rank)

	########Gather and calculate global mean#############################

        Collected_means=comm.gather(Local_mean,root=0)
	#Collected_means=np.asarray(Collected_means)

        if rank==0:
             Global_mean=[[]for i in range(No_Of_Clusters)]
             sub_group=[[]for i in range(len(Collected_means))]
	     #print('Collected_means',Collected_means)
	     ###############Calculate global means###########################
	     
             for j in range (No_Of_Clusters):
                     for i in range (len(Collected_means)):   
                         if  len(Collected_means[i][j])==0:
                                Collected_means[i][j]=[np.nan for k in range(No_Of_Features)]
                         else:    
		                  #This step because collected means is list of list
                                Collected_means[i][j]=Collected_means[i][j][0]
		         #print('collected means ',i,' ',j,'  ',Collected_means[i][j])
                         sub_group[i]=Collected_means[i][j]
                     #print('sub ',sub_group)
                     average=np.nanmean(sub_group,axis=0)
                     #print()
                     #print('averge ',average)
                     #if no instanse is assigned to a this cluster
                     if  math.isnan(average[0]):
                          average=np.random.randint(1,50,[No_Of_Features])
                     Global_mean[j]= average 
	      
		     #print('sub ggggg',sub_group)
                         
             #print('Global_means',np.shape(Global_mean))
             Cluster_centroids=Global_mean
             if w== Number_Of_Iterations-1:
                  End_Time=MPI.Wtime()
                  print('Time Collapsed',End_Time-Start_Time)
        Cluster_centroids=comm.bcast(Cluster_centroids,root=0)




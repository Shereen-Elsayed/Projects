import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import col,avg,udf,when,unix_timestamp
import pyspark.sql.functions  as func 
from pyspark.sql.types import DateType, TimestampType
from datetime import datetime,date
from pyspark.sql.functions import coalesce, to_date,stddev,mean,to_timestamp
from pyspark.sql.functions import collect_list, size,explode,countDistinct,count
from pyspark.sql.types import IntegerType
from pyspark.sql.types import ArrayType
from pyspark.sql import Row
import operator
from pyspark.sql.window import Window

conf = SparkConf().setAppName("Ex2").setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

df= sc.textFile('/host/HieldshiemMasters/Semester1/DistributedDataAnalytics/Exercises/Ex9_Solution/ml-10M100K/tags.dat').map(lambda x: x.split("::"))

df=df.toDF(['UserID','MovieID','Tag','Timestamp'])
df_Update =df.withColumn('time_datestring',func.from_unixtime('timestamp'))
df_Update =df_Update.withColumn('time_date',to_timestamp(df_Update.time_datestring, 'yyyy-MM-dd HH:mm:ss'))
#print(df_Update)
#df_Update.show()

#===== get all the time stamps for each user ========================
#test=df_Update.groupBy(['UserID'])
new=df_Update.groupBy(['UserID']).agg(collect_list("time_date"))
#test.show()           
#==========sort time stamps for each user===========================
#func=udf(lambda x:sorted(x.tolist()))

def sorter(l):
  res = sorted(l)
  return [item for item in res]

sort_udf = udf(sorter,ArrayType(TimestampType()))

sorted_df = new.withColumn('sorted', sort_udf(new['collect_list(time_date)'])) 
print(sorted_df )
#sorted_df.show () 
#====== get the difference between each two successive time stamps.
#step 1 get a column of previous value
#0 if y[0]  else (y[i]-y[i-1] for i in 

def diff(l):
  currsession=1
  output=list()
  for i in range(len(l)):
    if(i==0):
      output.append(currsession)
      continue
    
    diff=((l[i]-l[i-1]).seconds)/60
    if(diff>=30):
      currsession=currsession+1
    output.append(currsession)
    
  return [item for item in output]

diff_udf = udf(diff,ArrayType(IntegerType()))



sorted_df = sorted_df.withColumn("sessions",diff_udf(sorted_df['sorted']))
#sorted_df.show()
#===============================================================================
exploded_df=sorted_df.withColumn("sessionsid",explode(sorted_df['sessions']))
#exploded_df.show()
#out=exploded_df.where(col("UserID")=="46959")
#out.show()
#=========================Ex 2 calculate the frequencies =======================
freq = exploded_df.groupBy("UserID","sessionsid").agg(count("*").alias("frequency"))
out1=freq.where(col("UserID")=="1436")
#out1.show()
#freq.show()
#=======================Q3====================================================
freq2=freq.groupBy("UserID").agg(mean(col("frequency")))
freq22=freq.groupBy("UserID").agg(stddev(col("frequency").cast("double")))
out2=freq2.where(col("UserID")=="1436")
#out2.show()
out3=freq22.where(col("UserID")=="1436")
#out3.show()
#freq22.show()  
#freq2.show()
#=======================Q4====================================================
#freq_all = exploded_df.agg(count("*"))
#freq_all.show()
freq3=freq.agg(mean(col("frequency")))
freq4=freq.agg(stddev(col("frequency").cast("double")))
#freq3.show()
#freq4.show()
#=======================Q5====================================================
mean_all=freq3.select("avg(frequency)").rdd.flatMap(lambda x: x).collect()
std_all=freq4.select("stddev_samp(CAST(frequency AS DOUBLE))").rdd.flatMap(lambda x: x).collect()
filt=freq2.where((col("avg(frequency)")< mean_all[0]+2*std_all[0]) & (col("avg(frequency)")> mean_all[0]-2*std_all[0]))
filt.show()




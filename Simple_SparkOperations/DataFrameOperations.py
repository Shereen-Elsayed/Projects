import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
#def createDataFrame(rowRDD: RDD[Row], schema: StructType): DataFrame

conf = SparkConf().setAppName("Ex1").setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
a =sc.parallelize( [('spark'),('rdd'),('python'),('context'),('create'),('class')])
#a=a.toDF(['name','id'])


b = sc.parallelize([('operation'),('apache'),('scala'),('lambda'),('parallel'),('partition')])
#b=b.toDF(['name','id'])
a=a+b
#counts=a.flatMap(lambda a:[(c,1) for c in a]).reduceByKey(lambda x, y: x + y).collect()
counts=a.flatMap(lambda a:[(c,1) for c in a]).filter(lambda x: "s" in x).aggregateByKey(0,lambda x, y: x + y,lambda w, v: w + v).collect()
#result=a.join(b,a.name==b.name,"full_outer")
print(counts)

import os
import sys

os.environ['SPARK_HOME'] = 'C:\\Applications\\spark-1.6.1-bin-hadoop2.6'

sys.path.append("%s\\python\\lib\\py4j-0.9-src.zip" % os.getenv('SPARK_HOME'))
sys.path.append("%s\\python" % os.getenv('SPARK_HOME'))

try:
    import pyspark    
    print("Successfully imported Spark modules.")

except ImportError as e:
    print ("Could not import Spark modules", e)
    sys.exit(1)
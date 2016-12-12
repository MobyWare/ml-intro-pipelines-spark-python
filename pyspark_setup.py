## Got logic from this site here: http://renien.com/blog/accessing-pyspark-pycharm/
import os
import sys

sys.path.append("%s\\python\\lib\\py4j-0.10.3-src.zip" % os.getenv('SPARK_HOME'))
sys.path.append("%s\\python" % os.getenv('SPARK_HOME'))

try:
    import pyspark    
    print("Successfully imported Spark modules.")

except ImportError as e:
    print ("Could not import Spark modules", e)
    sys.exit(1)
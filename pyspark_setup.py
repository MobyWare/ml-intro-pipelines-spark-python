## Got logic from this site here: http://renien.com/blog/accessing-pyspark-pycharm/
import os
import sys

sys.path.append("%s\\python\\lib\\py4j-0.10.3-src.zip" % os.getenv('SPARK_HOME'))
sys.path.append("%s\\python" % os.getenv('SPARK_HOME'))

## Remove environment variables in my system that may mess with reading a local file in a python script

# Hadoop configs mess with reading relative file paths
if os.environ.get('HADOOP_CONF_DIR') is not None:
    del os.environ['HADOOP_CONF_DIR']


# Jupyter configs in my environment may mesh with running a python script. Removing PYSPARK_* assumes a default python interpreter in PATH
if os.environ.get('PYSPARK_PYTHON') is not None:
    del os.environ['PYSPARK_PYTHON']

if os.environ.get('PYSPARK_DRIVER_PYTHON') is not None:
    del os.environ['PYSPARK_DRIVER_PYTHON']

try:
    import pyspark    
    print("Successfully imported Spark modules.")

except ImportError as e:
    print ("Could not import Spark modules", e)
    sys.exit(1)
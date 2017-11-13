from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

def init():
    conf = SparkConf().setAppName("mainHousing").setMaster("local")
    return SparkContext(conf=conf)

def read_words(line):
    array = []
    word = ''
    iniciado = False
    for letter in line:
        if letter != ' ':
            iniciado = True
            word = word + letter
        elif iniciado:
            array.append(float(word))
            word = ''
            iniciado = False
    array.append(float(word))
    return array
                        

def main():
    print("hola")
    sc = init()
    rdd = sc.textFile("data/housing.data") #se convierte en rdd
    # rdd.foreach(lambda line: print(line))
    rddProcesado = rdd.map(lambda line: read_words(line))

    #pasando de RDD a Dataframe 
    spark = SparkSession(sc)
    headers = ["CRIM", "ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
    dataFrame = spark.createDataFrame(rddProcesado, headers)
    dataFrame.show()

    train_data, test_data = dataFrame.randomSplit([0.8,0.2], seed=12345)
    print("Longitud Training : {}".format(train_data.count()))
    print("Longitud Test : {}".format(test_data.count()))

    '''para mostrar ciertas columnas del dataframe'''
    #dataFrame.select("CRIM","MEDV").show() 

    #http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler

    assembler = VectorAssembler(inputCols=["CRIM"], outputCol="features") #combina las columnas inputCols en una columna outputCol
    data = assembler.transform(train_data) #transforma el train_data
    data.show()

    '''vectorassembler con select'''
    #data.select("features","MEDV").show()

    ''' REGRESION LINEAL '''
    #maxIter = maximo numero de iteraciones, regParam = para reducir el overfiting.
    lr = LinearRegression(maxIter=20,featuresCol="features",labelCol="MEDV",regParam=0.01)

    lr_model = lr.fit(data) #modelo con train_data
    #Para interpretacion: http://www.theanalysisfactor.com/interpreting-regression-coefficients/
    print("Coefficients: {}".format(lr_model.coefficients)) #es el parametro b (y = a + bx)
    print("Intercept: {}".format(lr_model.intercept)) #es el parametro a (y = a + bx)

    #Predecimos (predecimos con test data)
    t_data = assembler.transform(test_data)
    eval_data = lr_model.transform(t_data) #aca es la duda (transforma el dataset)
    
    #Realizamos una evaluacion
    evaluator = RegressionEvaluator(labelCol="MEDV", metricName= "rmse", predictionCol="prediction")
    print("RMSE: {}".format(evaluator.evaluate(eval_data)))

    ''' dibujamos la grafica '''
    arr_x = train_data.select("CRIM").rdd.collect()
    arr_y = train_data.select("MEDV").rdd.collect()

    plt.scatter(arr_x,arr_y)
    plt.xlabel("CRIM")
    plt.ylabel("MEDV")
    plt.show()

if __name__ == '__main__':
    main()
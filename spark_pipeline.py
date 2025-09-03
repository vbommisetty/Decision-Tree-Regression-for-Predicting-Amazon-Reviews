import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import PCA
from pyspark.ml.stat import Summarizer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics


# ---------- Begin definition of helper functions, if you need any ------------

# def task_1_helper():
#   pass

# -----------------------------------------------------------------------------


def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    df = product_data.join(review_data, 'asin', 'left')\
                     .select('asin', 'overall')\
                     .groupby('asin')\
                     .agg(F.avg('overall').alias('meanRating'), F.count('overall').alias('countRating'))
    df = df.replace({0: None}, subset=['countRating']).persist() #undo count putting 0
    
    
    count_total = df.count()
    mean_meanRating = df.select(F.avg('meanRating')).head()[0]
    variance_meanRating = df.select(F.variance('meanRating')).head()[0]
    numNulls_meanRating = df.where(df['meanRating'].isNull()).count()
    mean_countRating = df.select(F.avg('countRating')).head()[0]
    variance_countRating = df.select(F.variance('countRating')).head()[0]
    numNulls_countRating = df.where(df['countRating'].isNull()).count()




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }
    # Modify res:
    res['count_total'] = count_total
    res['mean_meanRating'] = mean_meanRating
    res['variance_meanRating'] = variance_meanRating
    res['numNulls_meanRating'] = numNulls_meanRating
    res['mean_countRating'] = mean_countRating
    res['variance_countRating'] = variance_countRating
    res['numNulls_countRating'] = numNulls_countRating



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------


def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    product_data = product_data.withColumn('category', F.flatten('categories')[0])
    product_data = product_data.replace({'': None}, subset=['category'])
    product_data = product_data.withColumn(
        bestSalesCategory_column,
        F.when(
            F.col(salesRank_column).isNotNull(),
            F.map_keys(F.col(salesRank_column))[0]
        ).otherwise(None)
    ).withColumn(
        bestSalesRank_column,
        F.when(
            F.col(salesRank_column).isNotNull(),
            F.map_values(F.col(salesRank_column))[0]
        ).otherwise(None)
    )

    count_total = product_data.count()
    mean_bestSalesRank = product_data.agg(F.avg(bestSalesRank_column)).first()[0]
    variance_bestSalesRank = product_data.agg(F.variance(bestSalesRank_column)).first()[0]
    numNulls_category = product_data.filter(F.col(category_column).isNull()).count()
    countDistinct_category = product_data.select(F.countDistinct('category')).head()[0]
    numNulls_bestSalesCategory = product_data.filter(F.col(bestSalesCategory_column).isNull()).count()
    countDistinct_bestSalesCategory = product_data.select(F.col(bestSalesCategory_column)).distinct().count() - 1 if numNulls_bestSalesCategory > 0 else product_data.select(F.col(bestSalesCategory_column)).distinct().count()





    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_bestSalesRank': None,
        'variance_bestSalesRank': None,
        'numNulls_category': None,
        'countDistinct_category': None,
        'numNulls_bestSalesCategory': None,
        'countDistinct_bestSalesCategory': None
    }
    # Modify res:
    res = {
        'count_total': count_total,
        'mean_bestSalesRank': mean_bestSalesRank,
        'variance_bestSalesRank': variance_bestSalesRank,
        'numNulls_category': numNulls_category,
        'countDistinct_category': countDistinct_category,
        'numNulls_bestSalesCategory': numNulls_bestSalesCategory,
        'countDistinct_bestSalesCategory': countDistinct_bestSalesCategory
    }



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------


def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    df_with_also_viewed = product_data.withColumn(
        attribute, 
        F.when(
            F.array_contains(F.map_keys(F.col(related_column)), attribute), 
            F.col(related_column).getItem(attribute)
        ).otherwise(None)
    )
    exploded_df = df_with_also_viewed.withColumn(attribute, F.explode_outer(F.col(attribute)))
    related_prods = exploded_df.select(asin_column, attribute)
    prices = product_data.select(F.col(asin_column).alias('pid'), price_column)
    joined = related_prods.join(prices, related_prods[attribute] == prices['pid'], how='left')
    grouped_by_prod = joined.groupBy(asin_column).agg(F.avg(F.col(price_column)).alias(meanPriceAlsoViewed_column))
    mean_meanPriceAlsoViewed, count_total, numNulls_meanPriceAlsoViewed, variance_meanPriceAlsoViewed = grouped_by_prod.select(
        F.avg(F.col(meanPriceAlsoViewed_column)).alias('mean'),
        F.count(F.col(asin_column)).alias('count'),
        F.count(F.when(F.isnull(F.col(meanPriceAlsoViewed_column)), 1)).alias('num_nulls'),
        F.variance(F.col(meanPriceAlsoViewed_column)).alias('variance')
    ).first()
    with_counts_also_viewed = df_with_also_viewed.withColumn(countAlsoViewed_column, F.size(F.col(attribute)))
    with_counts_also_viewed = with_counts_also_viewed.withColumn(
        countAlsoViewed_column, 
        F.when(
            (F.col(attribute).isNull()) | (F.size(F.col(attribute)) == 0), 
            None
        ).otherwise(F.size(F.col(attribute)))
    )
    mean_countAlsoViewed, variance_countAlsoViewed, numNulls_countAlsoViewed = with_counts_also_viewed.select(
        F.avg(F.col(countAlsoViewed_column)).alias('mean'),
        F.variance(F.col(countAlsoViewed_column)).alias('variance'),
        F.count(F.when(F.isnull(F.col(countAlsoViewed_column)), 1)).alias('num_nulls')
    ).first()




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanPriceAlsoViewed': None,
        'variance_meanPriceAlsoViewed': None,
        'numNulls_meanPriceAlsoViewed': None,
        'mean_countAlsoViewed': None,
        'variance_countAlsoViewed': None,
        'numNulls_countAlsoViewed': None
    }
    # Modify res:
    res['count_total'] = count_total
    res['mean_meanPriceAlsoViewed'] = mean_meanPriceAlsoViewed
    res['variance_meanPriceAlsoViewed'] = variance_meanPriceAlsoViewed
    res['numNulls_meanPriceAlsoViewed'] = numNulls_meanPriceAlsoViewed
    res['mean_countAlsoViewed'] = mean_countAlsoViewed
    res['variance_countAlsoViewed'] = variance_countAlsoViewed
    res['numNulls_countAlsoViewed'] = numNulls_countAlsoViewed



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------


def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    product_data = product_data.withColumn('price', product_data['price'].astype('float'))
    product_data = product_data.withColumn('meanImputedPrice', product_data['price'])\
                        .fillna(product_data.select(F.avg('price')).head()[0], subset=['meanImputedPrice'])
    product_data = product_data.withColumn('medianImputedPrice', product_data['price'])\
                        .fillna(product_data.approxQuantile("price", [0.5], 0.001)[0], subset=['medianImputedPrice'])
    product_data = product_data.withColumn('unknownImputedTitle', product_data['title'])\
                        .fillna('unknown', subset=['unknownImputedTitle'])
    
    
    count_total = product_data.count()
    mean_meanImputedPrice = product_data.select(F.avg('meanImputedPrice')).head()[0]
    variance_meanImputedPrice = product_data.select(F.variance('meanImputedPrice')).head()[0]
    numNulls_meanImputedPrice = product_data.where(F.col('meanImputedPrice').isNull()).count()
    mean_medianImputedPrice = product_data.select(F.avg('medianImputedPrice')).head()[0]
    variance_medianImputedPrice = product_data.select(F.variance('medianImputedPrice')).head()[0]
    numNulls_medianImputedPrice = product_data.where(F.col('medianImputedPrice').isNull()).count()
    numUnknowns_unknownImputedTitle = product_data.where(F.col('unknownImputedTitle')=='unknown').count()
    




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanImputedPrice': None,
        'variance_meanImputedPrice': None,
        'numNulls_meanImputedPrice': None,
        'mean_medianImputedPrice': None,
        'variance_medianImputedPrice': None,
        'numNulls_medianImputedPrice': None,
        'numUnknowns_unknownImputedTitle': None
    }
    # Modify res:
    res['count_total'] = count_total
    res['mean_meanImputedPrice'] = mean_meanImputedPrice
    res['variance_meanImputedPrice'] = variance_meanImputedPrice
    res['numNulls_meanImputedPrice'] = numNulls_meanImputedPrice
    res['mean_medianImputedPrice'] = mean_medianImputedPrice
    res['variance_medianImputedPrice'] = variance_medianImputedPrice
    res['numNulls_medianImputedPrice'] = numNulls_medianImputedPrice
    res['numUnknowns_unknownImputedTitle'] = numUnknowns_unknownImputedTitle



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------


def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    product_processed_data_output = product_processed_data.withColumn('titleArray', (F.split(F.lower(product_processed_data.title), ' ', -1)))
    model = M.feature.Word2Vec(vectorSize=16, seed=SEED, inputCol="titleArray", minCount=100, numPartitions= 4).fit(product_processed_data_output)




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': [(None, None), ],
        'word_1_synonyms': [(None, None), ],
        'word_2_synonyms': [(None, None), ]
    }
    # Modify res:
    res['count_total'] = product_processed_data_output.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------
    si = StringIndexer(inputCol='category', outputCol='si_output')
    ohe = OneHotEncoder(inputCol='si_output', outputCol='categoryOneHot', dropLast=False)
    pca = PCA(k=15, inputCol='categoryOneHot', outputCol='categoryPCA')
    pipe = Pipeline(stages=[si, ohe, pca])
    product_processed_data = pipe.fit(product_processed_data).transform(product_processed_data)
    
    count_total = product_processed_data.count()
    meanVector_categoryOneHot = product_processed_data.select(Summarizer.mean(product_processed_data.categoryOneHot)).head()[0]
    meanVector_categoryPCA = product_processed_data.select(Summarizer.mean(product_processed_data.categoryPCA)).head()[0]





    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:
    res['count_total'] = count_total
    res['meanVector_categoryOneHot'] = meanVector_categoryOneHot
    res['meanVector_categoryPCA'] = meanVector_categoryPCA



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------
    
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    dt = DecisionTreeRegressor(maxDepth=5, featuresCol='features', labelCol='overall', predictionCol='predDT')
    test_data = dt.fit(train_data).transform(test_data)
    
    evaluator = RegressionEvaluator(predictionCol='predDT', labelCol='overall', metricName='rmse')
    test_rmse = evaluator.evaluate(test_data)
    
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None
    }
    # Modify res:
    res['test_rmse'] = test_rmse

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    train_data, validation_data = train_data.randomSplit([0.75, 0.25])
    model1 = DecisionTreeRegressor(maxDepth=5, featuresCol='features', labelCol='overall', predictionCol='predDepth5').fit(train_data)
    model2 = DecisionTreeRegressor(maxDepth=7, featuresCol='features', labelCol='overall', predictionCol='predDepth7').fit(train_data)
    model3 = DecisionTreeRegressor(maxDepth=9, featuresCol='features', labelCol='overall', predictionCol='predDepth9').fit(train_data)
    model4 = DecisionTreeRegressor(maxDepth=12, featuresCol='features', labelCol='overall', predictionCol='predDepth12').fit(train_data)
    
    validation_data = model1.transform(validation_data)
    validation_data = model2.transform(validation_data)
    validation_data = model3.transform(validation_data)
    validation_data = model4.transform(validation_data)
    
    evaluator = RegressionEvaluator(labelCol='overall', metricName='rmse')
    validation_count = validation_data.count()
    valid_rmse_depth_5 = evaluator.evaluate(validation_data, {evaluator.predictionCol: 'predDepth5'})
    valid_rmse_depth_7 = evaluator.evaluate(validation_data, {evaluator.predictionCol: 'predDepth7'})
    valid_rmse_depth_9 = evaluator.evaluate(validation_data, {evaluator.predictionCol: 'predDepth9'})
    valid_rmse_depth_12 = evaluator.evaluate(validation_data, {evaluator.predictionCol: 'predDepth12'})
    
    
    predModel = [(model1, 'predDepth5'), (model2, 'predDepth7'), (model3, 'predDepth7'), (model4, 'predDepth12')]
    predRMSE = ['valid_rmse_depth_5', 'valid_rmse_depth_7', 'valid_rmse_depth_9', 'valid_rmse_depth_12']
    bestModel = min(zip(predModel, predRMSE), key=lambda x: x[1])[0]
    test_rmse = evaluator.evaluate(bestModel[0].transform(test_data), {evaluator.predictionCol: bestModel[1]})
    
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None,
        'valid_rmse_depth_5': None,
        'valid_rmse_depth_7': None,
        'valid_rmse_depth_9': None,
        'valid_rmse_depth_12': None,
    }
    # Modify res:
    res['test_rmse'] = test_rmse
    res['valid_rmse_depth_5'] = valid_rmse_depth_5
    res['valid_rmse_depth_7'] = valid_rmse_depth_7
    res['valid_rmse_depth_9'] = valid_rmse_depth_9
    res['valid_rmse_depth_12'] = valid_rmse_depth_12

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------
def task1_largest_discrepancy(flights_df, carriers_df):
    # Calculate the difference in minutes between scheduled and actual travel time
    flights_df = flights_df.withColumn(
        "ScheduledTravelTime",
        (F.unix_timestamp("ScheduledArrival") - F.unix_timestamp("ScheduledDeparture")) / 60
    ).withColumn(
        "ActualTravelTime",
        (F.unix_timestamp("ActualArrival") - F.unix_timestamp("ActualDeparture")) / 60
    ).withColumn(
        "TimeDiscrepancy",
        F.abs(F.col("ActualTravelTime") - F.col("ScheduledTravelTime"))
    )
    
    # Find the flights with the largest discrepancy
    window = Window.orderBy(F.desc("TimeDiscrepancy"))
    largest_discrepancy = flights_df.select("FlightNum", "CarrierCode", "Origin", "Destination", "TimeDiscrepancy") \
                                    .withColumn("Rank", F.row_number().over(window)) \
                                    .filter(F.col("Rank") <= 10)

    # Join with carriers to get carrier names
    largest_discrepancy = largest_discrepancy.join(
        carriers_df, flights_df.CarrierCode == carriers_df.CarrierCode, "left"
    ).select("FlightNum", "CarrierName", "Origin", "Destination", "TimeDiscrepancy")

    # Write to CSV
    largest_discrepancy.write.csv(task1_output, header=True)
    print(f"Task 1 output written to {task1_output}")

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    # Calculate departure delay
    flights_df = flights_df.withColumn(
        "DepartureDelay",
        (F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")) / 60
    )

    # Group by CarrierCode, calculate standard deviation, and filter airlines with more than 100 flights
    airline_delays = flights_df.groupBy("CarrierCode") \
                               .agg(
                                   F.count("FlightNum").alias("FlightCount"),
                                   F.stddev("DepartureDelay").alias("DelayStdDev")
                               ) \
                               .filter(F.col("FlightCount") > 100) \
                               .orderBy("DelayStdDev")

    # Join with carriers to get carrier names
    consistent_airlines = airline_delays.join(
        carriers_df, "CarrierCode", "left"
    ).select("CarrierName", "FlightCount", "DelayStdDev")

    # Write to CSV
    consistent_airlines.write.csv(task2_output, header=True)
    print(f"Task 2 output written to {task2_output}")

# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------
def task3_canceled_routes(flights_df, airports_df):
    # Calculate cancellation status and group by origin-destination pair
    flights_df = flights_df.withColumn(
        "Canceled", F.when(F.col("ActualDeparture").isNull(), 1).otherwise(0)
    )
    
    route_cancellations = flights_df.groupBy("Origin", "Destination") \
                                    .agg(
                                        F.count("FlightNum").alias("TotalFlights"),
                                        F.sum("Canceled").alias("TotalCanceled")
                                    ) \
                                    .withColumn("CancellationRate", F.col("TotalCanceled") / F.col("TotalFlights")) \
                                    .orderBy(F.desc("CancellationRate"))

    # Join with airports to get airport names
    canceled_routes = route_cancellations \
        .join(airports_df, route_cancellations.Origin == airports_df.AirportCode, "left") \
        .withColumnRenamed("AirportName", "OriginAirport") \
        .drop("AirportCode") \
        .join(airports_df, route_cancellations.Destination == airports_df.AirportCode, "left") \
        .withColumnRenamed("AirportName", "DestinationAirport") \
        .select("OriginAirport", "DestinationAirport", "CancellationRate")

    # Write to CSV
    canceled_routes.write.csv(task3_output, header=True)
    print(f"Task 3 output written to {task3_output}")

# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    # Create a time of day column
    flights_df = flights_df.withColumn(
        "DepartureHour", F.hour("ScheduledDeparture")
    ).withColumn(
        "TimeOfDay",
        F.when((F.col("DepartureHour") >= 6) & (F.col("DepartureHour") < 12), "Morning")
         .when((F.col("DepartureHour") >= 12) & (F.col("DepartureHour") < 18), "Afternoon")
         .when((F.col("DepartureHour") >= 18) & (F.col("DepartureHour") < 24), "Evening")
         .otherwise("Night")
    )

    # Calculate average delay by carrier and time of day
    carrier_performance = flights_df.withColumn(
        "DepartureDelay",
        (F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")) / 60
    ).groupBy("CarrierCode", "TimeOfDay") \
     .agg(F.avg("DepartureDelay").alias("AverageDelay")) \
     .orderBy("CarrierCode", "TimeOfDay")

    # Join with carriers to get carrier names
    carrier_performance_time_of_day = carrier_performance.join(
        carriers_df, "CarrierCode", "left"
    ).select("CarrierName", "TimeOfDay", "AverageDelay")

    # Write to CSV
    carrier_performance_time_of_day.write.csv(task4_output, header=True)
    print(f"Task 4 output written to {task4_output}")

# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()
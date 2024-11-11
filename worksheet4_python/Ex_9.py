import time
# Decorator to measure execution time
def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time
        print(f"Execution time for {func.__name__}: {execution_time:.6f} seconds")
        return result
    return wrapper
# Sample function that adds sleep for testing the execution time
@measure_execution_time
def add(a, b):
    time.sleep(2)  # Simulate a delay of 2 seconds
    return a + b
# Calling the sample function
add(6, 3)
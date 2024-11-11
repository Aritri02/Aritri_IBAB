import functools
# Logging decorator
def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Log function call with arguments and keyword arguments
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        # Log the return value
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper
# Example function using the decorator
@log_function_call
def add(a, b):
    return a + b
# Invoke the function
add(5, 3)
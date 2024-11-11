def division(a, b):
    try:
        # Try to perform the division
        result = a / b
    except ZeroDivisionError:
        # Handle division by zero
        print("Error: Division by zero is not allowed.")
        result = None
    except ValueError:
        # Handle the case where inputs cannot be converted to numbers
        print("Error: Invalid input value. Please provide numeric values.")
        result = None
    except Exception as e:
        # Handle any other unforeseen errors
        print(f"An unknown error occurred: {e}")
        result = None
    finally:
        # This block will always execute, regardless of whether an exception occurred
        print("Execution of the division operation has finished.")
        return result
# Example usage:
print(division(10, 5))  # Expected: 5.0
print(division(10, 0))  # Expected: Division by zero error
print(division("10", 2))  # Expected: Invalid input error
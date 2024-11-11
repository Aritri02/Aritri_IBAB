class FormulaError(Exception):
    """Custom exception for invalid formula input."""
    pass
def interactive_calculator():
    while True:
        # Get user input
        user_input = input("Enter a formula (e.g., '2 + 3') or 'quit' to exit: ")

        # Exit condition
        if user_input.lower() == 'quit':
            print("Exiting the calculator.")
            break
        # Split the input by spaces
        input_parts = user_input.split()
        try:
            # Check if the input consists of exactly 3 elements
            if len(input_parts) != 3:
                raise FormulaError("Formula must consist of two numbers and an operator.")
            # Extract the components: first number, operator, second number
            num1_str, operator, num2_str = input_parts
            # Try to convert both numbers to floats
            try:
                num1 = float(num1_str)
                num2 = float(num2_str)
            except ValueError:
                raise FormulaError("Both numbers must be valid numeric values.")
            # Check if the operator is valid
            if operator not in ('+', '-'):
                raise FormulaError("Operator must be either '+' or '-'.")
            # Perform the calculation
            if operator == '+':
                result = num1 + num2
            elif operator == '-':
                result = num1 - num2
            # Print the result
            print(f"Result: {result}")
        except FormulaError as fe:
            print(f"Error: {fe}")
# Run the interactive calculator
interactive_calculator()
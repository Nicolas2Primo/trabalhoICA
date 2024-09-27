from functools import wraps
from typing import Any
import time
import statistics


def timer(repetitions: int = 1) -> Any:
    def decorator(func: Any) -> Any:
        @wraps(func)
        def timer_wrapper(*args: tuple, **kwargs: dict[str, Any]) -> Any:
            execution_times = []
            
            for _ in range(repetitions):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                total_time = end_time - start_time
                execution_times.append(total_time)

            if repetitions > 1:
                avg_time = sum(execution_times) / repetitions
                stdev = statistics.stdev(execution_times)
                print(f'-----------------------------------------------------------------')
                print(f'[INFO] Function {func.__name__} executed {repetitions} times')
                print(f'[INFO] Total Time: {avg_time:.4f} ± {stdev:.4f} s')
                print(f'-----------------------------------------------------------------')
            else:
                print(f'-----------------------------------------------------------------')
                print(f'[INFO] Function {func.__name__} executed once')
                print(f'[INFO] Total Time: {execution_times[0]:.4f} s')
                print(f'-----------------------------------------------------------------')

            return result
        return timer_wrapper

    return decorator


# import sys
# from contextlib import redirect_stdout
# from io import StringIO

# def capture(filename):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             # Create a StringIO object to capture printed output
#             stdout_buffer = StringIO()

#             # Redirect stdout to the StringIO object
#             with redirect_stdout(stdout_buffer):
#                 # Execute the function and capture the result
#                 result = func(*args, **kwargs)

#             # Get the captured printed output
#             printed_output = stdout_buffer.getvalue()

#             # Save the captured output data and prints to a text file
#             with open(filename, 'w') as file:
#                 file.write("======================== Function Output ========================\n")
#                 file.write(str(result) + '\n')
#                 file.write("======================== Printed Output =========================\n")
#                 file.write(printed_output)

#             # Return the original function result
#             return result
#         return wrapper
#     return decorator

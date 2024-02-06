import subprocess

filename = 'results/even_odd_worklist_results.txt'

# Open the results file in write mode
with open(filename, 'w') as f:
    # Run your CUDA program 16 times
    for i in range(1, 18):
        input2 = i
        inputs = [1, input2, 4]

        # Convert inputs to string and join them with a newline to simulate pressing Enter
        inputs = '\n'.join(map(str, inputs))

        # Run your CUDA program
        process = subprocess.Popen(['./output'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Send the inputs to the program
        stdout, stderr = process.communicate(inputs)

        # Write the output to the file
        f.write(f"Run {i}:\n")
        f.write(stdout)
        f.write("--------------------\n")

print(f"Results written to {filename}")

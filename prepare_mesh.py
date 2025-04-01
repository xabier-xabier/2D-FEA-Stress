import csv
import re


def create_element():
    # Path to the CSV file
    input_file = 'element_input.csv'
    output_file = 'element_output.csv'

    # Expressions to extract the data from the CSV file
    regex_element = r'Element (\d+)'  # Extract Element ID
    regex_nodes = r'Nodes\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)'  # Extract nodes

    # Open the file and read it
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        
        # Open the output file to write it
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            
            # Write the first row (Description)
            writer.writerow(['Element Id', 'Node 1', 'Node 2', 'Node 3', 'Node 4'])
            
            # Read proccess teh data
            for line in reader:
                line = line[0]  # 1st column

                # Search the Element ID in the line
                element_match = re.search(regex_element, line)
                if element_match:
                    element_id = element_match.group(1)
                
                # Search for the nodes in the line
                nodes_match = re.search(regex_nodes, line)
                if nodes_match:
                    nodes = nodes_match.groups()
                    
                    # Write in the row with the Element ID and the nodes in the output file
                    writer.writerow([element_id] + list(nodes))

    print("The process has finished in:", output_file)
            
            
def create_node():
    
    # Define the regular expression to match the values you want to extract
    regex = r'^\s*(\d+)\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)'

    # Open the input CSV file and the output CSV file
    with open('entry_nodes.csv', 'r') as infile, open('output_nodes.csv', 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter=' ')  # Adjust the delimiter if needed (e.g., tab, spaces)
        writer = csv.writer(outfile)
        
        # Write headers to the output CSV
        writer.writerow(['Node ID', 'X Coordinate', 'Y Coordinate', 'DOF ID'])
        j=0
        # Loop through each row in the input CSV
        for row in reader:
            # Join the row into a single string with spaces to handle split columns
            row_string = ' '.join(row)
            
            # Search for the pattern using regex
            match = re.search(regex, row_string)
            if match:
                # Extract Node ID, X Coordinate, and Y Coordinate
                node_id = match.group(1)
                x_coord = match.group(2)
                y_coord = match.group(3)
                DOF_ID  = j
                
                # Write the result to the output CSV
                writer.writerow([node_id, x_coord, y_coord, DOF_ID])
                j+=1
            else:
                # If no match is found, print an error message
                print(f"No match found for row: {row_string}")
                
                
def node_variables():
    with open("output_nodes.csv", mode="r", encoding="utf-8") as out_nodes:
        reader=csv.reader(out_nodes)
        node_id, x_coord, y_coord, DOF_ID = [], [], [], []
        j=0
        
        for row in reader:
            if j!=0:
                node_id.append(row[0])
                x_coord.append(row[1])
                y_coord.append(row[2])
                DOF_ID.append(row[3])
                j+=1
            else:
                j+=1
                
    return node_id,x_coord,y_coord,DOF_ID

def element_variables():
    with open("element_output.csv", mode="r", encoding="utf-8") as file:
        reader=csv.reader(file)
        element_id, node_1, node_2, node_3, node_4= [], [], [], [], []
        j=0
        
        for row in reader:
            if j!=0:
                element_id.append(row[0])
                node_1.append(row[1])
                node_2.append(row[2])
                node_3.append(row[3])
                node_4.append(row[4])
                j+=1
            else:
                j+=1
    return element_id,node_1,node_2,node_3,node_4

def create_loads():
    # Define the regular expression to match the values you want to extract
    regex = r'^\s*(\d+)'

    # Open the input CSV file and the output CSV file
    with open('entry_loads.csv', 'r') as infile, open('output_loads.csv', 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter=' ')  # Adjust the delimiter if needed (e.g., tab, spaces)
        writer = csv.writer(outfile)
        
        # Write headers to the output CSV
        writer.writerow(['Node ID'])
        j=0
        # Loop through each row in the input CSV
        for row in reader:
            # Join the row into a single string with spaces to handle split columns
            row_string = ' '.join(row)
            
            # Search for the pattern using regex
            match = re.search(regex, row_string)
            if match:
                # Extract Node ID
                node_id = match.group(0)
                
                # Write the result to the output CSV
                writer.writerow([node_id])
                j+=1
            else:
                # If no match is found, print an error message
                print(f"No match found for row: {row_string}")
                
def node_loads():
    with open("output_loads.csv", mode="r", encoding="utf-8") as out_nodes:
        reader=csv.reader(out_nodes)
        node_id = []
        j=0
        
        for row in reader:
            if j!=0:
                node_id.append(row[0])
                j+=1
            else:
                j+=1
    return node_id

def create_restrictions():
    # Define the regular expression to match the values you want to extract
    regex = r'^\s*(\d+)'

    # Open the input CSV file and the output CSV file
    with open('entry_restrictions.csv', 'r') as infile, open('output_restrictions.csv', 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter=' ')  # Adjust the delimiter if needed (e.g., tab, spaces)
        writer = csv.writer(outfile)
        
        # Write headers to the output CSV
        writer.writerow(['Node ID'])
        j=0
        # Loop through each row in the input CSV
        for row in reader:
            # Join the row into a single string with spaces to handle split columns
            row_string = ' '.join(row)
            
            # Search for the pattern using regex
            match = re.search(regex, row_string)
            if match:
                # Extract Node ID
                node_id = match.group(0)
                
                # Write the result to the output CSV
                writer.writerow([node_id])
                j+=1
            else:
                # If no match is found, print an error message
                print(f"No match found for row: {row_string}")
                
def node_restrictions():
    with open("output_restrictions.csv", mode="r", encoding="utf-8") as out_nodes:
        reader=csv.reader(out_nodes)
        node_id = []
        j=0
        
        for row in reader:
            if j!=0:
                node_id.append(row[0])
                j+=1
            else:
                j+=1
    return node_id
    

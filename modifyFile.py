def modify_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        modified_line = ' '.join([item.split(':')[1] for item in line.split()])
        modified_lines.append(modified_line)

    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

# Call the function with the path to your file
# modify_file('Gunrockresults/kron_g500-logn16.txt')
# modify_file('Gunrockresults/chesapeake.txt')
# modify_file('Gunrockresults/rgg_n_2_16_s0.txt')
# modify_file('Gunrockresults/rgg_n_2_24_s0.txt')
# modify_file('Gunrockresults/inf-road_usa.txt')
# modify_file('Gunrockresults/inf-europe_osm.txt')
# modify_file('Gunrockresults/inf-luxembourg_osm.txt')
# modify_file('Gunrockresults/co-papers-dblp.txt')
# modify_file('Gunrockresults/co-papers-citeseer.txt')
# modify_file('Gunrockresults/hugetrace-00000.txt')
# modify_file('Gunrockresults/hugetrace-00020.txt')
# modify_file('Gunrockresults/delaunay_n17.txt')
# modify_file('Gunrockresults/delaunay_n23.txt')
# modify_file('Gunrockresults/delaunay_n24.txt')
modify_file('nlpkkt240.txt')
# modify_file('Gunrockresults/channel-500x100x100-b050.txt')

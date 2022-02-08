def remove_last_item_from_tuple_in_lists_in_dict_values(my_dict):
    """
    input format s.th. like
    my_dict = {0: [((0, -1), 0.85, 'removeme'),
               ((1, 0), -0.5, 'removeme'),
               ((2, -1), 0.7, 'removeme')],
           1: [((1, -1), 0.8, 'removeme'),
               ((2, 0), 0.7, 'removeme')],
           2: [((2, -1), 0.9, 'removeme')],
           3: [((3, -1), 0.8, 'removeme'),
               ((0, -2), 0.4, 'removeme')]}
    """
    for key in my_dict:
        my_list = my_dict[key]
        len_my_list = len(my_list)
        modified_list = []
        for list_index in range(len_my_list):
            my_tuple = my_list[list_index]
            modified_tuple = my_tuple[:-1]
            modified_list.append(modified_tuple)
        my_dict.update({key: modified_list})
def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    if input_length is None:
        return None
        assert padding in {'same', 'valid', 'full', 'causal'}
        dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding in ['same', 'causal']:
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride

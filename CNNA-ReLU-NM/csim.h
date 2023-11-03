#ifndef CNN_H
#define CNN_H

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

void train(int adapt, int network, int epoch, int print_to_mem, int print_start, int print_count);
void test(int adapt, int network);

void rewind(FILE *f);
float sigmoid(float x);
float sigmoid_deri(float y);
float relu(float x);
float relu_deri(float y);
void maxpool(float a, float b, float c, float d, float *max_num, int *max_index);
int max_out(float a[10]);
int max_out1to9(float a[9]);

void print_conv_wxb(int adapt);

void print_convSA_wxb(int adapt);

// void print_floatToHex(float number,  int print_to_stdout, const char * path, const char * mode, int print_to_file, int newline);

void print_MNIST_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int k_val);

void print_MNIST_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int k_val);

// sum_z = conv1_z + conva_z/convpa_z
// float conv_z[6][24][24]
void print_SA_sum_fmap_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float conv_z[6][24][24]
void print_SA_sum_fmap_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float conv_z_pa[6][28][28]
void print_PA_sum_fmap_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float conv_z_pa[6][28][28]
void print_PA_sum_fmap_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);


// conv1_z
// float conv1_z[6][24][24]
void print_SA_fmap_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float conv1_z[6][24][24]
void print_SA_fmap_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float conv1_z_pa[6][28][28]
void print_PA_fmap_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float conv1_z_pa[6][28][28]
void print_PA_fmap_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);



/// FMAP by H x W x Channels ///
// float conv1_z[6][24][24]
void print_SA_fmap_bychan_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float conv1_z[6][24][24]
void print_SA_fmap_bychan_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float conv1_z_pa[6][28][28]
void print_PA_fmap_bychan_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float conv1_z_pa[6][28][28]
void print_PA_fmap_bychan_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);



// Adapter_z
// float conva_z[6][24][24]
void print_SA_z_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float conva_z[6][24][24]
void print_SA_z_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float convpa_z[6][28][28]
void print_PA_z_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float convpa_z[6][28][28]
void print_PA_z_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);


// 
void print_SA_weights_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int k_val);

void print_SA_weights_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int k_val);

void print_SA_biases_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val);

void print_SA_biases_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val);


// 
void print_SA_delta_weights_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int k_val);

void print_SA_delta_weights_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int k_val);

void print_SA_delta_biases_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val);

void print_SA_delta_biases_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val);


// 
void print_PA_weights_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val);

void print_PA_weights_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val);

void print_PA_biases_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val);

void print_PA_biases_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val);


// 
void print_PA_delta_weights_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val);

void print_PA_delta_weights_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val);

void print_PA_delta_biases_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val);

void print_PA_delta_biases_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val);


// float deriv_actv[6][24][24]
void print_SA_deriv_actv_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);
void print_SA_deriv_actv_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float deriv_actv_pa[6][28][28]
void print_PA_deriv_actv_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);
void print_PA_deriv_actv_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float maxpool_error[6][24][24]
void print_SA_maxpool_error_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);
void print_SA_maxpool_error_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float maxpool_error_pa[6][28][28]
void print_PA_maxpool_error_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);
void print_PA_maxpool_error_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float hidden1ErrorSum[6][24][24]
void print_SA_delta_error_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);
void print_SA_delta_error_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

// float hidden1ErrorSum_pa[6][28][28]
void print_PA_delta_error_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);
void print_PA_delta_error_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val);

//mnist [60000][28][28]

/// Print parameters ///

// float conv_k_weights[6][5][5], 
void print_conv_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1, int dim2, int dim3);

// float conv_k_bias[6]
void print_conv_biases(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1);

// float conva_weight[6][6];
void print_SA_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1, int dim2);

// float conva_bias[6];
void print_SA_biases(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1);

// float convpa_weight[6];
void print_PA_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1);

// float convpa_bias[6];
void print_PA_biases(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1);

// max_to_hidden_weights[6][12][12][45] FC WEIGHT
void print_fc_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1, int dim2, int dim3, int dim4);

// max_to_hidden_weights_pa[6][14][14][45] FC WEIGHT
void print_fcPA_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1, int dim2, int dim3, int dim4);

// hidden_bias[45]
void print_fc_biases(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1);

// hidden_to_out_weights[45][10]
void print_out_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1, int dim2);

// out_bias[10]
void print_out_biases(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1);

// hidden_to_out1to9_weights[45][9]
void print_out1to9_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1, int dim2);

// out_bias1to9[9]
void print_out1to9_biases(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1);

void load_param_conv_weights(const char * path);
void load_param_conv_biases(const char * path);
void load_param_SA_weights(const char * path);
void load_param_SA_biases(const char * path);
void load_param_PA_weights(const char * path);
void load_param_PA_biases(const char * path);
void load_param_fc_weights(const char * path);
// max_to_hidden_weights_pa[6][14][14][45] FC WEIGHT
void load_param_fcpa_weights(const char * path);
void load_param_fc_biases(const char * path);
void load_param_out_weights(const char * path);
void load_param_out_biases(const char * path);
void load_param_out1to9_weights(const char * path);
void load_param_out1to9_biases(const char * path);

#endif
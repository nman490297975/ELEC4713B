// SS-CNNA in C based off code by Jiong Si:
//Source: https://digitalscholarship.unlv.edu/thesesdissertations/3845/
//Description: Code that designs logic of Convolutional Neural Network with 1 convolutional, max-pooling and fully connected layer

//Code snippet for reading mnist-data:
//Source: https://github.com/AndrewCarterUK/mnist-neural-network-plain-c/tree/master
//Description: This code snippet reads the mnist data files into custom types.

// Main logic by Niyana Yohnnaes
// Print/load/save logic, design including cross-entropy and debugging by Nicholas Manning 

// USYD Code Citation
//
// I, Nicholas Manning, acknowledge that I was inspired by tutorials, forum posts and learning resources in my writing of the code // for loading, saving, printing data from this C simulator.


#include "csim.h"
#include "mnist.h"
#include "param_read.h"
#include "string.h"

union Float32Union {
    char bytes[4];
    float float32;
};

/**
 * Downloaded from: http://yann.lecun.com/exdb/mnist/
 */
const char * train_images_file = "data/train-images.idx3-ubyte";
const char * train_labels_file = "data/train-labels.idx1-ubyte";
const char * test_images_file = "data/t10k-images.idx3-ubyte";
const char * test_labels_file = "data/t10k-labels.idx1-ubyte";

mnist_dataset_t * train_dataset, * test_dataset;
mnist_dataset_t_fp * train_dataset_fp, * test_dataset_fp;
mnist_image_t_fp * images_fp, *images_test_fp;
float * labels_fp, *labels_test_fp;

float mnist_1to9_train_images[54077][784];
float mnist_1to9_train_labels[54077];
float mnist_1to9_test_images[9020][784];
float mnist_1to9_test_labels[9020];

int count_test;

int pool_index[6][12][12], pool_index_pa[6][14][14];

float learning_rate = 0.01;
float image[28][28];

float conv_k_weights[6][5][5], max_to_hidden_weights[6][12][12][45], max_to_hidden_weights_pa[6][14][14][45], hidden_to_out_weights[45][10], hidden_to_out_weights1to9[45][9];
float conv_k_dw[6][5][5], max_to_hidden_dw[6][12][12][45], max_to_hidden_dw_pa[6][14][14][45], hidden_to_out_dw[45][10], hidden_to_out_dw1to9[45][9];

float deriv_actv[6][24][24]; 		//NM
float deriv_actv_pa[6][28][28]; 	//NM

float conv_k_bias[6], hidden_bias[45], out_bias[10], out_bias1to9[9];
float conv_k_db[6], hidden_db[45], out_db[10], out_db1to9[9];

float conv_z[6][24][24], conv_actv[6][24][24], conv_z_pa[6][28][28], conv_actv_pa[6][28][28];
float maxpool_z[6][12][12], maxpool_actv[6][12][12], maxpool_z_pa[6][14][14], maxpool_actv_pa[6][14][14];
float hidden_z[45], hidden_actv[45];
float out_z[10], out_actv[10], out_z1to9[9], out_actv1to9[9];
unsigned char actual_output[10], actual_output1to9[9];

float outError[10], outError1to9[9];
float temp3ErrorSum, temp2ErrorSum, temp1ErrorSum, temp0ErrorSum, tempErrorSum00;
float hidden3ErrorSum[45], hidden2ErrorSum[6][12][12], hidden2ErrorSum_pa[6][14][14],  hidden1ErrorSum[6][24][24], hidden1ErrorSum_pa[6][28][28], maxpool_error[6][24][24], maxpool_error_pa[6][28][28];

float conv1_z[6][24][24], conv1_z_pa[6][28][28];

float conva_z[6][24][24];
float conva_weight[6][6]; //[6][6][1][1]
float conva_bias[6]; //[6][1][1]
float conva_dw[6][6]; //[6][1][1]
float conva_db[6]; //[6][1][1]

float convpa_z[6][28][28];
float convpa_weight[6]; //[6][6][1][1]
float convpa_bias[6]; //[6][1][1]
float convpa_dw[6]; //[6][1][1]
float convpa_db[6]; //[6][1][1]

int main()
{	

	//////////////////////////////////////Convert to Single Precision Floating Point////////////////////////////////////
    // Read the datasets from the files
    train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
    test_dataset = mnist_get_dataset(test_images_file, test_labels_file);

    train_dataset_fp = calloc(1, sizeof(mnist_dataset_t_fp));
    images_fp = malloc(train_dataset->size * sizeof(mnist_image_t_fp));
    labels_fp = malloc(train_dataset->size * sizeof(float));

    train_dataset_fp->size = train_dataset->size;
    int mnist_count = 0;
    for (int j = 0; j < train_dataset_fp->size; j++) {
    	labels_fp[j] = (float)train_dataset->labels[j];
        for (int i=0; i<784; i++) {
            images_fp[j].pixels[i] = (float)train_dataset->images[j].pixels[i];
            images_fp[j].pixels[i] = images_fp[j].pixels[i] / (float)255;
            if (labels_fp[j] != 0.0)
            	mnist_1to9_train_images[mnist_count][i] = images_fp[j].pixels[i];
        }
	    if (labels_fp[j] != 0.0){
        	mnist_1to9_train_labels[mnist_count] = labels_fp[j];
        	mnist_count++;
	    }
    }

    train_dataset_fp->images = images_fp; 
    train_dataset_fp-> labels = labels_fp;


    test_dataset_fp = calloc(1, sizeof(mnist_dataset_t_fp));
    images_test_fp = malloc(test_dataset->size * sizeof(mnist_image_t_fp));
    labels_test_fp = malloc(test_dataset->size * sizeof(float));

    test_dataset_fp->size = test_dataset->size;
    mnist_count = 0;
    for (int j = 0; j < test_dataset_fp->size; j++) {
    	labels_test_fp[j] = (float)test_dataset->labels[j];
        for (int i=0; i<784; i++) {
            images_test_fp[j].pixels[i] = (float)test_dataset->images[j].pixels[i];
            images_test_fp[j].pixels[i] = images_test_fp[j].pixels[i] / (float)255;
            if (labels_test_fp[j] != 0)
            	mnist_1to9_test_images[mnist_count][i] = images_test_fp[j].pixels[i];
        }
        if (labels_test_fp[j] != 0) {
        	mnist_1to9_test_labels[mnist_count] = labels_test_fp[j];
        	mnist_count++;
        }
    }

    test_dataset_fp->images = images_test_fp; 
    test_dataset_fp-> labels = labels_test_fp;
	//////////////////////////////////////Convert to Single Precision Floating Point////////////////////////////////////



	/////////////////////////////////////Initialize Weights/////////////////////////////////////////
	for (int i = 0; i < 6; i++)
		for (int j = 0; j < 5; j++)
			for (int k = 0; k < 5; k++)
				conv_k_weights[i][j][k] = rand() % 10000 / 10000.0 - 0.5;
	
	for (int i = 0; i < 6; i++)
		for (int m = 0; m < 12; m++)
			for (int n = 0; n < 12; n++)
				for (int j = 0; j < 45; j++)
					max_to_hidden_weights[i][m][n][j] = rand() % 10000 / 10000.0 - 0.5;
	for (int i = 0; i < 6; i++)
		for (int m = 0; m < 14; m++)
			for (int n = 0; n < 14; n++)
				for (int j = 0; j < 45; j++)
					max_to_hidden_weights_pa[i][m][n][j] = rand() % 10000 / 10000.0 - 0.5;
	
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
			conva_weight[i][j] = rand() % 10000 / 10000.0 - 0.5;
		conva_bias[i] = rand() % 10000 / 10000.0 - 0.5;
	}
	for (int i = 0; i < 6; i++)
	{
		convpa_weight[i] = rand() % 10000 / 10000.0 - 0.5;
		convpa_bias[i] = rand() % 10000 / 10000.0 - 0.5;
	}

	
	for (int i = 0; i < 45; i++)
		for (int j = 0; j < 10; j++)
			hidden_to_out_weights[i][j] = rand() % 10000 / 10000.0 - 0.5;
	for (int i = 0; i < 45; i++)
		for (int j = 0; j < 9; j++)
			hidden_to_out_weights1to9[i][j] = rand() % 10000 / 10000.0 - 0.5;
	
	for (int i = 0; i < 6; i++)
		conv_k_bias[i] = rand() % 10000 / 10000.0 - 0.5;
	
	for (int i = 0; i < 45; i++)
		out_bias[i] = rand() % 10000 / 10000.0 - 0.5;
	
	for (int i = 0; i < 10; i++)
		out_bias[i] = rand() % 10000 / 10000.0 - 0.5;
	for (int i = 0; i < 9; i++)
		out_bias1to9[i] = rand() % 10000 / 10000.0 - 0.5;
	/////////////////////////////////////Initialize Weights/////////////////////////////////////////
	
	
	///////////////////////////////////////////////////////////Training and Testing////////////////////////////////////////////////////////////////



	/// Remove mem files ///
	printf("\n\n/// Removing files ///\n\n");
	remove("rtlmem/mnist_hex.mem");
	remove("rtlmem/mnist_float.mem");

	remove("rtlmem/sa.conv1.fmap_sum_hex.mem");
	remove("rtlmem/sa.conv1.fmap_sum_float.mem");	
	remove("rtlmem/pa.conv1.fmap_sum_hex.mem");
	remove("rtlmem/pa.conv1.fmap_sum_float.mem");

	remove("rtlmem/sa.conv1.fmap_bychan_hex.mem");
	remove("rtlmem/sa.conv1.fmap_bychan_float.mem");
	remove("rtlmem/pa.conv1.fmap_sum_bychan_hex.mem");
	remove("rtlmem/pa.conv1.fmap_sum_bychan_float.mem");

	remove("rtlmem/sa.conv1.fmap_hex.mem");
	remove("rtlmem/sa.conv1.fmap_float.mem");	
	remove("rtlmem/pa.conv1.fmap_hex.mem");
	remove("rtlmem/pa.conv1.fmap_float.mem");

	remove("rtlmem/sa.weights_hex.mem");
	remove("rtlmem/sa.weights_float.mem");
	remove("rtlmem/pa.weights_hex.mem");
	remove("rtlmem/pa.weights_float.mem");

	remove("rtlmem/sa.biases_hex.mem");
	remove("rtlmem/sa.biases_float.mem");
	remove("rtlmem/pa.biases_hex.mem");
	remove("rtlmem/pa.biases_float.mem");

	remove("rtlmem/sa.sa.z_hex.mem");
	remove("rtlmem/sa.sa.z_float.mem");
	remove("rtlmem/pa.pa.z_hex.mem");
	remove("rtlmem/pa.pa.z_float.mem");

	remove("rtlmem/sa.delta_weights_hex.mem");
	remove("rtlmem/sa.delta_weights_float.mem");
	remove("rtlmem/pa.delta_weights_hex.mem");
	remove("rtlmem/pa.delta_weights_float.mem");

	remove("rtlmem/sa.delta_biases_hex.mem");
	remove("rtlmem/sa.delta_biases_float.mem");
	remove("rtlmem/pa.delta_biases_hex.mem");
	remove("rtlmem/pa.delta_biases_float.mem");

	remove("rtlmem/sa.deriv_actv_hex.mem");
	remove("rtlmem/sa.deriv_actv_float.mem");
	remove("rtlmem/pa.deriv_actv_hex.mem");
	remove("rtlmem/pa.deriv_actv_float.mem");

	remove("rtlmem/sa.maxpool_error_hex.mem");
	remove("rtlmem/sa.maxpool_error_float.mem");	
	remove("rtlmem/pa.maxpool_error_hex.mem");
	remove("rtlmem/pa.maxpool_error_float.mem");	

	remove("rtlmem/sa.delta_error_hex.mem");
	remove("rtlmem/sa.delta_error_float.mem");
	remove("rtlmem/pa.delta_error_hex.mem");
	remove("rtlmem/pa.delta_error_float.mem");

	/// ///




	int adapt = 0;

	//Train original network 1to9
	// for (int epoch = 0; epoch < 1; epoch++)
	// {
	// 	printf("\n//////////////EPOCH %d (Train)//////////////////\n", epoch+1);

	// 	//Train original 1to9 network Without Adapter
	// 	adapt = 0;
	// 	train(adapt, 0, epoch, 0, 0, 0);

	// 	printf("End of Epoch %d\n", epoch + 1);
	// }
	// //Testing original network 1to9
	// printf("Beginning of Testing\n");
	// adapt = 0;
	// test(adapt, 0);

	// //Print Convolution Weights and Biases aftern original network
	//print_conv_wxb();

	//Serial Adapt on MNIST0toA9
	for (int epoch = 0; epoch < 1; epoch++)
	{
		printf("\n//////////////EPOCH %d (Adapt)//////////////////\n", epoch+1);

		//Train network 0to9 With Adapter
		adapt = 1;
		// print flag == 1, start index == 0, end_index == 9, produces 10 samples
		train(adapt, 1, epoch, 1, 0, 9); 

		printf("End of Epoch %d\n", epoch + 1);
	}
	// //Testing adapted network 0to9
	// printf("Beginning of Testing\n");
	// adapt = 1;
	// test(adapt, 1);

	// //Print Convolution Weights and Biases aftern Serial Adapter network
	//print_conv_wxb();

	// //Parallel Adapt on MNIST0toA9
	// for (int epoch = 0; epoch < 6; epoch++)
	// {
	// 	printf("\n//////////////EPOCH %d (Adapt)//////////////////\n", epoch+1);

	// 	//Train network 0to9 With Adapter
	// 	adapt = 2;
	// 	train(adapt, 1, epoch, 1, 0, 10); 

	// 	printf("End of Epoch %d\n", epoch + 1);
	// }
	// //Testing adapted network 0to9
	// printf("Beginning of Testing\n");
	// adapt = 2;
	// test(adapt, 1);

	// //Print Convolution Weights and Biases aftern Serial Adapter network
	//print_conv_wxb();

	// //Finetune without adapter
	// for (int epoch = 0; epoch < 3; epoch++)
	// {
	// 	printf("\n//////////////EPOCH %d (Adapt)//////////////////\n", epoch+1);

	// 	//Train network 0to9 With Adapter
	// 	adapt = 0;
	// 	train(adapt, 1, epoch, 1, 0, 10); 

	// 	printf("End of Epoch %d\n", epoch + 1);
	// }
	// //Testing adapted network 0to9
	// printf("Beginning of Testing\n");
	// adapt = 0;
	// test(adapt, 1);

	// //Print Convolution Weights and Biases aftern Serial Adapter network
	//print_conv_wxb();

}

void train (int adapt, int network, int epoch, int print_to_mem, int print_start, int print_end) //Network (0 = 1to9) (1 = 0to9) (2 = fine_tune 1to9)
{
	int count_train = 0;

	//////////Getting Network Size///////////
	int network_size = 0;
	if (network == 0) //1to9
		network_size = 54077;
	else //0to10
		network_size = train_dataset_fp->size;
	//////////Getting Network Size//////////

	int network_print_start_index = 0;
	int network_print_end_index = 0;

	/// Print files to mem flags ///
	if(print_to_mem == 1)
	{
		network_print_start_index = print_start;
		network_print_end_index = print_end;
	}
	/// ///


	for (int k = 0; k < network_size ; k++)
	{
		/////////////////////////Feed image into image array////////////////////////////
	    int pixelIndex = 0;

	    for(int j=0;j<MNIST_IMAGE_WIDTH;j++)
	    {
	        for(int l=0;l<MNIST_IMAGE_HEIGHT;l++)
	        {
	        	if (network == 0)
	        		image[j][l] = mnist_1to9_train_images[k][pixelIndex];
	            else
	            	image[j][l] = train_dataset_fp->images[k].pixels[pixelIndex];
	            pixelIndex++;
	            //printf("%3d ",(int)(image[j][l]*255));
	        }
	        //printf("\n");
	    }

	    /// TAP MNIST TO MEM ///
	    if(print_to_mem == 1 && k >= network_print_start_index && k < network_print_end_index) 
		{
			print_MNIST_hex(adapt, "mnist hex", 1, "rtlmem/mnist_hex.mem", "a", 1, 10, 28, 28, k);
			print_MNIST_float(adapt, "mnist float", 1, "rtlmem/mnist_float.mem", "a", 1, 10, 28, 28, k);
			printf("\n\n\n/// k %d MNIST mem printed ///\n\n\n", k);
		} 		
		else if (print_to_mem == 1 && k >= network_print_start_index && k == network_print_end_index) 
		{
			print_MNIST_hex(adapt, "mnist hex", 1, "rtlmem/mnist_hex.mem", "a", 0, 10, 28, 28, k);
			print_MNIST_float(adapt, "mnist float", 1, "rtlmem/mnist_float.mem", "a", 0, 10, 28, 28, k);
			printf("\n\n\n/// k %d MNIST mem printed done ///\n\n\n", k);
		}
	    /// ///

	    int label = 0;
	    if (network == 0){
	    	label = mnist_1to9_train_labels[k];
	    	//printf("Label: %f\n",mnist_1to9_train_labels[k]);
	    	for (int j = 1; j < 10; j++) {
				if (j == label) 
					actual_output1to9[j-1] = 1;
				else 
					actual_output1to9[j-1] = 0;
				//printf("%d: %d\n", j-1, actual_output1to9[j-1]);
			}
	    }
	    else {
	    	label = train_dataset_fp->labels[k];
	    	//printf("Label: %f\n",train_dataset_fp->labels[k]);
	    	for (int j = 0; j < 10; j++) {
				if (j == label) 
					actual_output[j] = 1;
				else 
					actual_output[j] = 0;
				//printf("%d: %d\n", j, actual_output[j]);
			}
	    }


		/////////////////////////Feed image into image array////////////////////////////
		

		/////////////////////////Forward Propoagation///////////////////////////////////
		//Convolutional Layer
		for (int i = 0; i < 6; i++)
			for (int m = 0; m < 24; m++)
				for (int n = 0; n < 24; n++)
				{

					if (adapt == 0)
						conv_z[i][m][n] = 0.0;
					else
						conv1_z[i][m][n] = conv_k_bias[i];

					for (int j = 0; j < 5; j++)
					for (int k = 0; k < 5; k++)
					{
						if (adapt == 0)
							conv_z[i][m][n] = conv_z[i][m][n] + image[j + m][k + n] * conv_k_weights[i][j][k];
						else 
							conv1_z[i][m][n] = conv1_z[i][m][n] + image[j + m][k + n] * conv_k_weights[i][j][k];		
					}

					if (adapt == 0)
						conv_actv[i][m][n] = relu(conv_z[i][m][n] + conv_k_bias[i]); //sigmoid(conv_z[i][m][n] + conv_k_bias[i]);
				}
		//If Parallel Adapter, 2 Padding
		if (adapt == 2) {
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 28; m++)
			        for (int n = 0; n < 28; n++)
			            conv1_z_pa[i][m][n] = 0.0;
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 24; m++)
			        for (int n = 0; n < 24; n++)
			            conv1_z_pa[i][m + 4 / 2][n + 4 / 2] = conv1_z[i][m][n];
		}

		//Adapter Layer
		//Serial Adapter Layer
		if (adapt == 1)
		{
			for (int i = 0; i < 6; i++) //To get six out channels
				for (int m = 0; m < 24; m++)
					for (int n = 0; n < 24; n++)
					{
						//Add a bias
						conva_z[i][m][n] = conva_bias[i];
						for (int a = 0 ; a < 6; a++) //Point-wise Convolution
							conva_z[i][m][n] = conva_z[i][m][n] + conv1_z[a][m][n] * conva_weight[i][a];
					}

			//Add up Conv Layer and Serial Adapter Layer and do Activiation
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 24; m++)
					for (int n = 0; n < 24; n++)
					{
						conv_z[i][m][n] = conv1_z[i][m][n] + conva_z[i][m][n];
						conv_actv[i][m][n] = relu(conv_z[i][m][n]); //sigmoid(conv_z[i][m][n]);
					}

			/// SA MEM ///
			int dimn1 = 6;
			int dimn2 = 24;
			int dimn3 = 24;			
			if(print_to_mem == 1 && k >= network_print_start_index && k < network_print_end_index) 
			{
				print_SA_sum_fmap_hex(adapt, "SA fmap_sum", 1, "rtlmem/sa.conv1.fmap_sum_hex.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);
				print_SA_sum_fmap_float(adapt, "SA fmap_sum float", 1, "rtlmem/sa.conv1.fmap_sum_float.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);

				print_SA_fmap_hex(adapt, "SA fmap", 1, "rtlmem/sa.conv1.fmap_hex.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);
				print_SA_fmap_float(adapt, "SA fmap float", 1, "rtlmem/sa.conv1.fmap_float.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);

				print_SA_fmap_bychan_hex(adapt, "SA fmap bychan", 1, "rtlmem/sa.conv1.fmap_bychan_hex.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);
				print_SA_fmap_bychan_float(adapt, "SA fmap bychan float", 1, "rtlmem/sa.conv1.fmap_bychan_float.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);

				print_SA_z_hex(adapt, "SA z", 1, "rtlmem/sa.sa.z_hex.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);
				print_SA_z_float(adapt, "SA z float", 1, "rtlmem/sa.sa.z_float.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);

				print_SA_weights_hex(adapt, "sa weights hex", 1, "rtlmem/sa.weights_hex.mem", "a", 1, 10, dimn1, 6, k);
				print_SA_weights_float(adapt, "sa weights float", 1, "rtlmem/sa.weights_float.mem", "a", 1, 10, dimn1, 6, k);

				print_SA_biases_hex(adapt, "sa biases hex", 1, "rtlmem/sa.biases_hex.mem", "a", 1, 10, dimn1, k);
				print_SA_biases_float(adapt, "sa biases float", 1, "rtlmem/sa.biases_float.mem", "a", 1, 10, dimn1, k);
				printf("\n\n\n/// k %d SA mem printed ///\n\n\n", k);
			} 
			// Don't print newline at end
			else if (print_to_mem == 1 && k >= network_print_start_index && k == network_print_end_index) 
			{
				print_SA_sum_fmap_hex(adapt, "SA fmap_sum", 1, "rtlmem/sa.conv1.fmap_sum_hex.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);
				print_SA_sum_fmap_float(adapt, "SA fmap_sum float", 1, "rtlmem/sa.conv1.fmap_sum_float.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);

				print_SA_fmap_hex(adapt, "SA fmap", 1, "rtlmem/sa.conv1.fmap_hex.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);
				print_SA_fmap_float(adapt, "SA fmap float", 1, "rtlmem/sa.conv1.fmap_float.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);

				print_SA_fmap_bychan_hex(adapt, "SA fmap bychan", 1, "rtlmem/sa.conv1.fmap_bychan_hex.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);
				print_SA_fmap_bychan_float(adapt, "SA fmap bychan float", 1, "rtlmem/sa.conv1.fmap_bychan_float.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);

				print_SA_z_hex(adapt, "SA z", 1, "rtlmem/sa.sa.z_hex.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);
				print_SA_z_float(adapt, "SA z float", 1, "rtlmem/sa.sa.z_float.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);

				print_SA_weights_hex(adapt, "sa weights hex", 1, "rtlmem/sa.weights_hex.mem", "a", 0, 10, dimn1, 6, k);
				print_SA_weights_float(adapt, "sa weights float", 1, "rtlmem/sa.weights_float.mem", "a", 0, 10, dimn1, 6, k);

				print_SA_biases_hex(adapt, "sa biases hex", 1, "rtlmem/sa.biases_hex.mem", "a", 0, 10, dimn1, k);
				print_SA_biases_float(adapt, "sa biases float", 1, "rtlmem/sa.biases_float.mem", "a", 0, 10, dimn1, k);
				printf("\n\n\n/// k %d SA mem printed done ///\n\n\n", k);
			}									
			/// ///
		}
		//Parallel Adapter
		else if (adapt == 2) {
			for (int i = 0; i < 6; i++) //To get six out channels
				for (int m = 0; m < 28; m++)
					for (int n = 0; n < 28; n++)
						convpa_z[i][m][n] = convpa_bias[i] + image[m][n] * convpa_weight[i]; 

			//Add up Conv Layer and Parallel Adapter Layer and do Activiation
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 28; m++)
					for (int n = 0; n < 28; n++)
					{
						conv_z_pa[i][m][n] = conv1_z_pa[i][m][n] + convpa_z[i][m][n];
						conv_actv_pa[i][m][n] = relu(conv_z_pa[i][m][n]); //sigmoid(conv_z_pa[i][m][n]);
					}

			/// PA MEM ///	
			int dimn1 = 6;
			int dimn2 = 28;
			int dimn3 = 28;		
			if(print_to_mem == 1 && k >= network_print_start_index && k < network_print_end_index) 
			{
				print_PA_sum_fmap_hex(adapt, "PA fmap_sum", 1, "rtlmem/pa.conv1.fmap_sum_hex.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);
				print_PA_sum_fmap_float(adapt, "PA fmap_sum float", 1, "rtlmem/pa.conv1.fmap_sum_float.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);

				print_PA_fmap_hex(adapt, "PA fmap", 1, "rtlmem/pa.conv1.fmap_hex.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);
				print_PA_fmap_float(adapt, "PA fmap float", 1, "rtlmem/pa.conv1.fmap_float.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);

				print_PA_fmap_bychan_hex(adapt, "PA fmap bychan", 1, "rtlmem/pa.conv1.fmap_bychan_hex.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);
				print_PA_fmap_bychan_float(adapt, "PA fmap bychan float", 1, "rtlmem/pa.conv1.fmap_bychan_float.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);

				print_PA_z_hex(adapt, "PA z", 1, "rtlmem/pa.pa.z_hex.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);
				print_PA_z_float(adapt, "PA z float", 1, "rtlmem/pa.pa.z_float.mem", "a", 1, 10, dimn1, dimn2, dimn3, k);

				print_PA_weights_hex(adapt, "pa weights", 1, "rtlmem/pa.weights_hex.mem", "a", 1, 10, dimn1, k);
				print_PA_weights_float(adapt, "pa weights", 1, "rtlmem/pa.weights_float.mem", "a", 1, 10, dimn1, k);
				print_PA_biases_hex(adapt, "pa biases", 1, "rtlmem/pa.biases_hex.mem", "a", 1, 10, dimn1, k);
				print_PA_biases_float(adapt, "pa biases", 1, "rtlmem/pa.biases_float.mem", "a", 1, 10, dimn1, k);
				printf("\n\n\n/// k %d PA mem printed ///\n\n\n", k);

			} 
			// Don't print newline for last one
			else if (print_to_mem == 1 && k >= network_print_start_index && k == network_print_end_index) 
			{
				print_PA_sum_fmap_hex(adapt, "PA fmap_sum", 1, "rtlmem/pa.conv1.fmap_sum_hex.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);
				print_PA_sum_fmap_float(adapt, "PA fmap_sum float", 1, "rtlmem/pa.conv1.fmap_sum_float.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);

				print_PA_fmap_hex(adapt, "PA fmap", 1, "rtlmem/pa.conv1.fmap_hex.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);
				print_PA_fmap_float(adapt, "PA fmap float", 1, "rtlmem/pa.conv1.fmap_float.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);

				print_PA_fmap_bychan_hex(adapt, "PA fmap bychan", 1, "rtlmem/pa.conv1.fmap_bychan_hex.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);
				print_PA_fmap_bychan_float(adapt, "PA fmap bychan float", 1, "rtlmem/pa.conv1.fmap_bychan_float.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);

				print_PA_z_hex(adapt, "PA z", 1, "rtlmem/pa.pa.z_hex.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);
				print_PA_z_float(adapt, "PA z float", 1, "rtlmem/pa.pa.z_float.mem", "a", 0, 10, dimn1, dimn2, dimn3, k);

				print_PA_weights_hex(adapt, "pa weights", 1, "rtlmem/pa.weights_hex.mem", "a", 0, 10, dimn1, k);
				print_PA_weights_float(adapt, "pa weights", 1, "rtlmem/pa.weights_float.mem", "a", 0, 10, dimn1, k);
				print_PA_biases_hex(adapt, "pa biases", 1, "rtlmem/pa.biases_hex.mem", "a", 0, 10, dimn1, k);
				print_PA_biases_float(adapt, "pa biases", 1, "rtlmem/pa.biases_float.mem", "a", 0, 10, dimn1, k);
				printf("\n\n\n/// k %d PA mem printed done ///\n\n\n", k);
			}									
			
			/// ///
		}

		//Maxpooling Layer
		//Parallel Adapter
		if (adapt == 2) {
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 14; m++)
					for (int n = 0; n < 14; n++)
					{
						maxpool(conv_actv_pa[i][2 * m][2 * n], conv_actv_pa[i][2 * m][2 * n + 1], conv_actv_pa[i][2 * m + 1][2 * n], conv_actv_pa[i][2 * m + 1][2 * n + 1], &maxpool_z_pa[i][m][n], &pool_index_pa[i][m][n]);
						maxpool_actv_pa[i][m][n] = maxpool_z_pa[i][m][n];
					}
		}
		else {
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 12; m++)
					for (int n = 0; n < 12; n++)
					{
						maxpool(conv_actv[i][2 * m][2 * n], conv_actv[i][2 * m][2 * n + 1], conv_actv[i][2 * m + 1][2 * n], conv_actv[i][2 * m + 1][2 * n + 1], &maxpool_z[i][m][n], &pool_index[i][m][n]);
						maxpool_actv[i][m][n] = maxpool_z[i][m][n];
					}
		}

		//FC Hidden Layer
		if (adapt == 2) {
			for (int k = 0; k < 45; k++)
			{
				hidden_z[k] = 0.0;
				for (int i = 0; i < 6; i++)
				for (int m = 0; m < 14; m++)
				for (int n = 0; n < 14; n++)
				hidden_z[k] = hidden_z[k] + maxpool_actv_pa[i][m][n] * max_to_hidden_weights_pa[i][m][n][k];
				hidden_actv[k] = relu(hidden_z[k] + hidden_bias[k]); //sigmoid(hidden_z[k] + hidden_bias[k]);
			}
		}
		else {
			for (int k = 0; k < 45; k++)
			{
				hidden_z[k] = 0.0;
				for (int i = 0; i < 6; i++)
				for (int m = 0; m < 12; m++)
				for (int n = 0; n < 12; n++)
				hidden_z[k] = hidden_z[k] + maxpool_actv[i][m][n] * max_to_hidden_weights[i][m][n][k];
				hidden_actv[k] = relu(hidden_z[k] + hidden_bias[k]); //sigmoid(hidden_z[k] + hidden_bias[k]);
			}
		}

		//FC Output Layer
		if (network == 0)
		{
			for (int i = 0; i < 9; i++)
			{
				out_z1to9[i] = 0.0;
				for (int j = 0; j < 45; j++)
					out_z1to9[i] = out_z1to9[i] + hidden_actv[j] * hidden_to_out_weights1to9[j][i];

				out_actv1to9[i] = sigmoid(out_z1to9[i] + out_bias1to9[i]);
				outError1to9[i] = out_actv1to9[i] - actual_output1to9[i]; //Fundamental Equation 1 --> Output error = (aL - y) 
			}
		}
		else
		{
			for (int i = 0; i < 10; i++)
			{
				out_z[i] = 0.0;
				for (int j = 0; j < 45; j++)
					out_z[i] = out_z[i] + hidden_actv[j] * hidden_to_out_weights[j][i];

				out_actv[i] = sigmoid(out_z[i] + out_bias[i]);
				outError[i] = out_actv[i] - actual_output[i]; //Fundamental Equation 1 --> Output error = (aL - y) 
			}
		}
		/////////////////////////Forward Propoagation///////////////////////////////////



		/////////////////////////Backwards Propoagation///////////////////////////////////
		//FC Output Layer
		if (network == 0)
		{
			for (int i = 0; i < 9; i++)
			{
				out_db1to9[i] = outError1to9[i]; //Fundamental Equation 3 --> db = error 
			}
			for (int i = 0; i < 45; i++)
				for (int j = 0; j < 9; j++)
					hidden_to_out_dw1to9[i][j] = outError1to9[j] * hidden_actv[i]; //Fundamental Equation 4 --> dw = a_in x error_out
		}
		else
		{
			for (int i = 0; i < 10; i++)
			{
				out_db[i] = outError[i]; //Fundamental Equation 3 --> db = error 
			}
			for (int i = 0; i < 45; i++)
				for (int j = 0; j < 10; j++)
					hidden_to_out_dw[i][j] = outError[j] * hidden_actv[i]; //Fundamental Equation 4 --> dw = a_in x error_out
		}

		//Fully Connected Hidden Layer
		if (network == 0)
		{
			for (int i = 0; i < 45; i++)
			{
				temp3ErrorSum = 0.0;
				for (int j = 0; j < 9; j++)
					temp3ErrorSum = temp3ErrorSum + outError1to9[j] * hidden_to_out_weights1to9[i][j]; //Partial of Fundamental Equation 2 --> Hidden error = w_l+1 x error_l+1
				
				hidden3ErrorSum[i] = temp3ErrorSum * relu_deri(hidden_z[i]); //sigmoid_deri(hidden_z[i]); // Partial of Fundamental Equation 2 --> Hidden error = (w_l+1 x error_l+1) x dactv(z)
				hidden_db[i] = hidden3ErrorSum[i]; //Fundamental Equation 3 --> db = error
			}
		}
		else
		{
			for (int i = 0; i < 45; i++)
			{
				temp3ErrorSum = 0.0;
				for (int j = 0; j < 10; j++)
					temp3ErrorSum = temp3ErrorSum + outError[j] * hidden_to_out_weights[i][j]; //Partial of Fundamental Equation 2 --> Hidden error = w_l+1 x error_l+1
				
				hidden3ErrorSum[i] = temp3ErrorSum * relu_deri(hidden_z[i]); //sigmoid_deri(hidden_z[i]); // Partial of Fundamental Equation 2 --> Hidden error = (w_l+1 x error_l+1) x dactv(z)
				hidden_db[i] = hidden3ErrorSum[i]; //Fundamental Equation 3 --> db = error
			}
		}
		
		if (adapt == 2) {
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 14; m++)
					for (int n = 0; n < 14; n++)
						for (int j = 0; j < 45; j++)
							max_to_hidden_dw_pa[i][m][n][j] = hidden3ErrorSum[j] *	maxpool_actv_pa[i][m][n]; //Fundamental Equation 4 --> dw = a_in x error_out
		}
		else {	
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 12; m++)
					for (int n = 0; n < 12; n++)
						for (int j = 0; j < 45; j++)
							max_to_hidden_dw[i][m][n][j] = hidden3ErrorSum[j] *	maxpool_actv[i][m][n]; //Fundamental Equation 4 --> dw = a_in x error_out
		}

		//Convolutional Layer Error
		if (adapt == 2){
			int index;
			for (int i = 0; i < 6; i++)
			{
				temp1ErrorSum = 0.0;
				index = 0;
				for (int j = 0; j < 28; j++)
					for (int k = 0; k < 28; k++)
					{

						temp0ErrorSum = 0.0;
						int maxIndex = pool_index_pa[i][j / 2][k / 2];

						if (j % 2 == 0) {
							// Values from 0 to 23 alternate between 0 and 1
						    index = (k+j) % 2;
						} else {
							// Values greater than or equal to 24 alternate between 2 and 3
						    index = ((k+j) % 2 == 0) ? 2 : 3;
						 }

						for (int m = 0; m < 45; m++) //Output Layer
						{
							temp0ErrorSum = temp0ErrorSum + hidden3ErrorSum[m] * max_to_hidden_weights_pa[i][j][k][m]; //Partial of Fundamental Equation 2 --> Hidden error = w_l+1 x error_l+1
						}

						//If a_jk is max in 2x2 window
						if(index == maxIndex) {
							maxpool_error_pa[i][j][k] = temp0ErrorSum;
							deriv_actv_pa[i][j][k] = relu_deri(conv_z_pa[i][j][k]); //NM

							hidden1ErrorSum_pa[i][j][k] = temp0ErrorSum * relu_deri(conv_z_pa[i][j][k]); //sigmoid_deri(conv_actv_pa[i][j][k]); // Partial of Fundamental Equation 2 --> Hidden error = (w_l+1 x error_l+1) x dactv(z)
						}
						//Else error is equal to 0
						else {
							maxpool_error_pa[i][j][k] = 0.0;
							hidden1ErrorSum_pa[i][j][k] = 0.0;
						}

						temp1ErrorSum = temp1ErrorSum + hidden1ErrorSum_pa[i][j][k];
					}

				convpa_db[i] = temp1ErrorSum; //Fundamental Equation 3 --> db = error
			}
		}
		else {
			int index;
			for (int i = 0; i < 6; i++)
			{
				temp1ErrorSum = 0.0;
				index = 0;
				for (int j = 0; j < 24; j++)
					for (int k = 0; k < 24; k++)
					{

						temp0ErrorSum = 0.0;
						int maxIndex = pool_index[i][j / 2][k / 2];

						if (j % 2 == 0) {
							// Values from 0 to 23 alternate between 0 and 1
						    index = (k+j) % 2;
						} else {
							// Values greater than or equal to 24 alternate between 2 and 3
						    index = ((k+j) % 2 == 0) ? 2 : 3;
						 }

						for (int m = 0; m < 45; m++) //Output Layer
						{
							temp0ErrorSum = temp0ErrorSum + hidden3ErrorSum[m] * max_to_hidden_weights[i][j][k][m]; //Partial of Fundamental Equation 2 --> Hidden error = w_l+1 x error_l+1
						}

						//If a_jk is max in 2x2 window
						if(index == maxIndex) {
							maxpool_error[i][j][k] = temp0ErrorSum;
							deriv_actv[i][j][k] = relu_deri(conv_z[i][j][k]); //NM

							hidden1ErrorSum[i][j][k] = temp0ErrorSum * relu_deri(conv_z[i][j][k]); //sigmoid_deri(conv_actv[i][j][k]); // Partial of Fundamental Equation 2 --> Hidden error = (w_l+1 x error_l+1) x dactv(z)
						}
						//Else error is equal to 0
						else {
							maxpool_error[i][j][k] = 0.0;
							hidden1ErrorSum[i][j][k] = 0.0;
						}

						temp1ErrorSum = temp1ErrorSum + hidden1ErrorSum[i][j][k];
					}

				//Adapter Backprop 
				if (adapt == 0)
					conv_k_db[i] = temp1ErrorSum; //Fundamental Equation 3 --> db = error
				else
					conva_db[i] = temp1ErrorSum; //Fundamental Equation 3 --> db = error
			}
		}

		if (adapt == 0)
		{
			//Convolutional Layer
			for (int i = 0; i < 6; i++)
				for (int j = 0; j < 5; j++)
					for (int k = 0; k < 5; k++)
					{
						float tempErrorSum = 0.0;
						for (int m = 0; m < 24; m++)
							for (int n = 0; n < 24; n++)
							{
							tempErrorSum = tempErrorSum +
							hidden1ErrorSum[i][m][n] * image[m + j][n + k];
							}

						conv_k_dw[i][j][k] = tempErrorSum;
					}
		}
		else if (adapt == 1)
		{
			for (int i = 0; i < 6; i ++)
			{
				float tempErrorSum = 0.0;
				float tempErrorSum2 = 0.0;
				float tempErrorSum3 = 0.0;
				float tempErrorSum4 = 0.0;
				float tempErrorSum5 = 0.0;
				float tempErrorSum6 = 0.0;
				for (int m = 0; m < 24; m++)
					for (int n = 0; n < 24; n++)
					{
						tempErrorSum = tempErrorSum + hidden1ErrorSum[i][m][n] * conv1_z[0][m][n];
						tempErrorSum2 = tempErrorSum2 + hidden1ErrorSum[i][m][n] * conv1_z[1][m][n];
						tempErrorSum3 = tempErrorSum3 + hidden1ErrorSum[i][m][n] * conv1_z[2][m][n];
						tempErrorSum4 = tempErrorSum4 + hidden1ErrorSum[i][m][n] * conv1_z[3][m][n];
						tempErrorSum5 = tempErrorSum5 + hidden1ErrorSum[i][m][n] * conv1_z[4][m][n];
						tempErrorSum6 = tempErrorSum6 + hidden1ErrorSum[i][m][n] * conv1_z[5][m][n];
					}
				conva_dw[i][0] = tempErrorSum;
				conva_dw[i][1] = tempErrorSum2;
				conva_dw[i][2] = tempErrorSum3;
				conva_dw[i][3] = tempErrorSum4;
				conva_dw[i][4] = tempErrorSum5;
				conva_dw[i][5] = tempErrorSum6;
			}
		}
		else
		{
			for (int i = 0; i < 6; i++)
			{
				float tempErrorSum = 0.0;
				for (int m = 0; m < 28; m++)
					for (int n = 0; n < 28; n++)
					{
						tempErrorSum = tempErrorSum +
							hidden1ErrorSum_pa[i][m][n] * image[m][n];
					}
				convpa_dw[i] = tempErrorSum;
			}
		}


		// Backprop prints //
		int dimn1 = 6;

		if(print_to_mem == 1 && k >= network_print_start_index && k < network_print_end_index) 
		{
			print_SA_delta_error_hex(adapt, "SA delta_error", 1, "rtlmem/sa.delta_error_hex.mem", "a", 1, 10, dimn1, 24, 24, k);
			print_SA_delta_error_float(adapt, "SA delta_error", 1, "rtlmem/sa.delta_error_float.mem", "a", 1, 10, dimn1, 24, 24, k);
			print_PA_delta_error_hex(adapt, "PA delta_error", 1, "rtlmem/pa.delta_error_hex.mem", "a", 1, 10, dimn1, 24, 24, k);
			print_PA_delta_error_float(adapt, "PA delta_error", 1, "rtlmem/pa.delta_error_float.mem", "a", 1, 10, dimn1, 24, 24, k);

			print_SA_deriv_actv_hex(adapt, "SA deriv_actv", 1, "rtlmem/sa.deriv_actv_hex.mem", "a", 1, 10, dimn1, 24, 24, k);
			print_SA_deriv_actv_float(adapt, "SA deriv_actv", 1, "rtlmem/sa.deriv_actv_float.mem", "a", 1, 10, dimn1, 24, 24, k);
			print_PA_deriv_actv_hex(adapt, "PA deriv_actv", 1, "rtlmem/pa.deriv_actv_hex.mem", "a", 1, 10, dimn1, 28, 28, k);
			print_PA_deriv_actv_float(adapt, "PA deriv_actv", 1, "rtlmem/pa.deriv_actv_float.mem", "a", 1, 10, dimn1, 28, 28, k);
			
			print_SA_maxpool_error_hex(adapt, "SA maxpool_error", 1, "rtlmem/sa.maxpool_error_hex.mem", "a", 1, 10, dimn1, 24, 24, k);
			print_SA_maxpool_error_float(adapt, "SA maxpool_error", 1, "rtlmem/sa.maxpool_error_float.mem", "a", 1, 10, dimn1, 24, 24, k);
			print_PA_maxpool_error_hex(adapt, "PA maxpool_error", 1, "rtlmem/pa.maxpool_error_hex.mem", "a", 1, 10, dimn1, 28, 28, k);
			print_PA_maxpool_error_float(adapt, "PA maxpool_error", 1, "rtlmem/pa.maxpool_error_float.mem", "a", 1, 10, dimn1, 28, 28, k);	

			print_SA_delta_weights_hex(adapt, "SA delta_weights", 1, "rtlmem/sa.delta_weights_hex.mem", "a", 1, 10, dimn1, 6, k);
			print_SA_delta_weights_float(adapt, "SA delta_weights", 1, "rtlmem/sa.delta_weights_float.mem", "a", 1, 10, dimn1, 6, k);		
			print_PA_delta_weights_hex(adapt, "PA delta_weights", 1, "rtlmem/pa.delta_weights_hex.mem", "a", 1, 10, dimn1, k);
			print_PA_delta_weights_float(adapt, "PA delta_weights", 1, "rtlmem/pa.delta_weights_float.mem", "a", 1, 10, dimn1, k);		
			
			print_SA_delta_biases_hex(adapt, "SA delta_biases", 1, "rtlmem/sa.delta_biases_hex.mem", "a", 1, 10, dimn1, k);
			print_SA_delta_biases_float(adapt, "SA delta_biases", 1, "rtlmem/sa.delta_biases_float.mem", "a", 1, 10, dimn1, k);
			print_PA_delta_weights_hex(adapt, "PA delta_biases", 1, "rtlmem/pa.delta_biases_hex.mem", "a", 1, 10, dimn1, k);
			print_PA_delta_weights_float(adapt, "PA delta_biases", 1, "rtlmem/pa.delta_biases_float.mem", "a", 1, 10, dimn1, k);

			printf("\n\n\n/// k %d BP mem printed ///\n\n\n", k);

		} 
		// Don't print newline for last one
		else if (print_to_mem == 1 && k >= network_print_start_index && k == network_print_end_index) 
		{
			print_SA_delta_error_hex(adapt, "SA delta_error", 1, "rtlmem/sa.delta_error_hex.mem", "a", 0, 10, dimn1, 24, 24, k);
			print_SA_delta_error_float(adapt, "SA delta_error", 1, "rtlmem/sa.delta_error_float.mem", "a", 0, 10, dimn1, 24, 24, k);
			print_PA_delta_error_hex(adapt, "PA delta_error", 1, "rtlmem/pa.delta_error_hex.mem", "a", 0, 10, dimn1, 24, 24, k);
			print_PA_delta_error_float(adapt, "PA delta_error", 1, "rtlmem/pa.delta_error_float.mem", "a", 0, 10, dimn1, 24, 24, k);

			print_SA_deriv_actv_hex(adapt, "SA deriv_actv", 1, "rtlmem/sa.deriv_actv_hex.mem", "a", 0, 10, dimn1, 24, 24, k);
			print_SA_deriv_actv_float(adapt, "SA deriv_actv", 1, "rtlmem/sa.deriv_actv_float.mem", "a", 0, 10, dimn1, 24, 24, k);
			print_PA_deriv_actv_hex(adapt, "PA deriv_actv", 1, "rtlmem/pa.deriv_actv_hex.mem", "a", 0, 10, dimn1, 28, 28, k);
			print_PA_deriv_actv_float(adapt, "PA deriv_actv", 1, "rtlmem/pa.deriv_actv_float.mem", "a", 0, 10, dimn1, 28, 28, k);
			
			print_SA_maxpool_error_hex(adapt, "SA maxpool_error", 1, "rtlmem/sa.maxpool_error_hex.mem", "a", 0, 10, dimn1, 24, 24, k);
			print_SA_maxpool_error_float(adapt, "SA maxpool_error", 1, "rtlmem/sa.maxpool_error_float.mem", "a", 0, 10, dimn1, 24, 24, k);
			print_PA_maxpool_error_hex(adapt, "PA maxpool_error", 1, "rtlmem/pa.maxpool_error_hex.mem", "a", 0, 10, dimn1, 28, 28, k);
			print_PA_maxpool_error_float(adapt, "PA maxpool_error", 1, "rtlmem/pa.maxpool_error_float.mem", "a", 0, 10, dimn1, 28, 28, k);	

			print_SA_delta_weights_hex(adapt, "SA delta_weights", 1, "rtlmem/sa.delta_weights_hex.mem", "a", 0, 10, dimn1, 6, k);
			print_SA_delta_weights_float(adapt, "SA delta_weights", 1, "rtlmem/sa.delta_weights_float.mem", "a", 0, 10, dimn1, 6, k);		
			print_PA_delta_weights_hex(adapt, "PA delta_weights", 1, "rtlmem/pa.delta_weights_hex.mem", "a", 0, 10, dimn1, k);
			print_PA_delta_weights_float(adapt, "PA delta_weights", 1, "rtlmem/pa.delta_weights_float.mem", "a", 0, 10, dimn1, k);		
			print_SA_delta_biases_hex(adapt, "SA delta_biases", 1, "rtlmem/sa.delta_biases_hex.mem", "a", 0, 10, dimn1, k);
			print_SA_delta_biases_float(adapt, "SA delta_biases", 1, "rtlmem/sa.delta_biases_float.mem", "a", 0, 10, dimn1, k);
			print_PA_delta_weights_hex(adapt, "PA delta_biases", 1, "rtlmem/pa.delta_biases_hex.mem", "a", 0, 10, dimn1, k);
			print_PA_delta_weights_float(adapt, "PA delta_biases", 1, "rtlmem/pa.delta_biases_float.mem", "a", 0, 10, dimn1, k);
			
			printf("\n\n\n/// k %d BP mem printed done ///\n\n\n", k);
		}
		/////////////////////////Backwards Propoagation///////////////////////////////////


		/////////////////////////Update Weights and Biases///////////////////////////////////
		//Output Layer
		if (network == 0)
		{
			for(int i=0; i < 9; i++)
				out_bias1to9[i] = out_bias1to9[i] - learning_rate * out_db1to9[i];

			for (int i = 0; i < 45; i++)
				for (int j = 0; j < 9; j++)
					hidden_to_out_weights1to9[i][j] = hidden_to_out_weights1to9[i][j] - learning_rate * hidden_to_out_dw1to9[i][j];
		}
		else
		{
			for(int i=0; i < 10; i++)
				out_bias[i] = out_bias[i] - learning_rate * out_db[i];

			for (int i = 0; i < 45; i++)
				for (int j = 0; j < 10; j++)
					hidden_to_out_weights[i][j] = hidden_to_out_weights[i][j] - learning_rate * hidden_to_out_dw[i][j];
		}

		//Hidden Layer
		for (int i = 0; i < 45; i++)
			hidden_bias[i] = hidden_bias[i] - learning_rate * hidden_db[i];

		if (adapt == 2)
		{
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 14; m++)
					for (int n = 0; n < 14; n++)
						for (int j = 0; j < 45; j++)
							max_to_hidden_weights_pa[i][m][n][j] = max_to_hidden_weights_pa[i][m][n][j] - learning_rate * max_to_hidden_dw_pa[i][m][n][j];
		}
		else
		{
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 12; m++)
					for (int n = 0; n < 12; n++)
						for (int j = 0; j < 45; j++)
							max_to_hidden_weights[i][m][n][j] = max_to_hidden_weights[i][m][n][j] - learning_rate * max_to_hidden_dw[i][m][n][j];
		}

		//Convolutional Kernel
		if (adapt == 0)
		{
			for (int i = 0; i < 6; i++)
				conv_k_bias[i] = conv_k_bias[i] - learning_rate * conv_k_db[i];

			for (int i = 0; i < 6; i++)
				for (int j = 0; j < 5; j++)
					for (int k = 0; k < 5; k++)
						conv_k_weights[i][j][k] =  conv_k_weights[i][j][k] - learning_rate * conv_k_dw[i][j][k];
		}
		else if (adapt == 1)
		{
			for (int i = 0; i < 6; i++)
			{
				for (int j = 0; j < 6; j++)
					conva_weight[i][j] = conva_weight[i][j] - learning_rate * conva_dw[i][j];

				conva_bias[i] = conva_bias[i] - learning_rate * conva_db[i];
			}

		}
		else
		{
			for (int i = 0; i < 6; i++)
			{
				convpa_weight[i] = convpa_weight[i] - learning_rate * convpa_dw[i];
				convpa_bias[i] = convpa_bias[i] - learning_rate * convpa_db[i];
			}
		}
		/////////////////////////Update Weights and Biases///////////////////////////////////
		

		////////////////////////////////Display progress///////////////////////////////////
		if(network == 0)
		{
			if ( (max_out1to9(out_actv1to9) + 1) != label)
				count_train++;
		}
		else {
			if (max_out(out_actv) != label)
				count_train++;
		}

		double progress = (((double)k + 1) / (double)network_size) * 100.0;
	    printf("TRAINING: reading image %5d / %5d progress [%3d%%]  ", k + 1, network_size, (int)progress);

	    float accuracy = (1.0 - ((float)count_train / (float)(k + 1))) * 100.0;
	    printf("RESULTS: correct=%5d  incorrect=%5d  accuracy=%5.4f%% \n", k + 1 - count_train, count_train, accuracy);
	    ////////////////////////////////Display progress///////////////////////////////////

	}
}

void test(int adapt, int network) 
{
	int count_test = 0;


	//////////Getting Network Size///////////
	int network_size = 0;
	if (network == 0) //1to9
		network_size = 9020;
	else //0to10
		network_size = test_dataset_fp->size;
	//////////Getting Network Size//////////


	for (int k = 0; k < network_size ; k++)
	{
		/////////////////////////Feed image into image array////////////////////////////
	    int pixelIndex = 0;

	    for(int j=0;j<MNIST_IMAGE_WIDTH;j++)
	    {
	        for(int l=0;l<MNIST_IMAGE_HEIGHT;l++)
	        {
	        	if (network == 0)
	        		image[j][l] = mnist_1to9_test_images[k][pixelIndex];
	            else
	            	image[j][l] = test_dataset_fp->images[k].pixels[pixelIndex];
	            pixelIndex++;
	            //printf("%3d ",(int)(image[j][l]*255));
	        }
	        //printf("\n");
	    }

	    int label = 0;
	    if (network == 0){
	    	label = mnist_1to9_test_labels[k];
	    	//printf("Label: %f\n",mnist_1to9_train_labels[k]);
	    	for (int j = 1; j < 10; j++) {
				if (j == label) 
					actual_output1to9[j-1] = 1;
				else 
					actual_output1to9[j-1] = 0;
				//printf("%d: %d\n", j-1, actual_output1to9[j-1]);
			}
	    }
	    else {
	    	label = test_dataset_fp->labels[k];
	    	//printf("Label: %f\n",train_dataset_fp->labels[k]);
	    	for (int j = 0; j < 10; j++) {
				if (j == label) 
					actual_output[j] = 1;
				else 
					actual_output[j] = 0;
				//printf("%d: %d\n", j, actual_output[j]);
			}
	    }


		/////////////////////////Feed image into image array////////////////////////////
		

		/////////////////////////Forward Propoagation///////////////////////////////////
		//Convolutional Layer
		for (int i = 0; i < 6; i++)
			for (int m = 0; m < 24; m++)
				for (int n = 0; n < 24; n++)
				{

					if (adapt == 0)
						conv_z[i][m][n] = 0.0;
					else
						conv1_z[i][m][n] = conv_k_bias[i];

					for (int j = 0; j < 5; j++)
					for (int k = 0; k < 5; k++)
					{
						if (adapt == 0)
							conv_z[i][m][n] = conv_z[i][m][n] + image[j + m][k + n] * conv_k_weights[i][j][k];
						else 
							conv1_z[i][m][n] = conv1_z[i][m][n] + image[j + m][k + n] * conv_k_weights[i][j][k];		
					}

					if (adapt == 0)
						conv_actv[i][m][n] = relu(conv_z[i][m][n] + conv_k_bias[i]); //sigmoid(conv_z[i][m][n] + conv_k_bias[i]);
				}
		//If Parallel Adapter, 2 Padding
		if (adapt == 2) {
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 28; m++)
			        for (int n = 0; n < 28; n++)
			            conv1_z_pa[i][m][n] = 0.0;
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 24; m++)
			        for (int n = 0; n < 24; n++)
			            conv1_z_pa[i][m + 4 / 2][n + 4 / 2] = conv1_z[i][m][n];
		}

		//Adapter Layer
		//Serial Adapter Layer
		if (adapt == 1)
		{
			for (int i = 0; i < 6; i++) //To get six out channels
				for (int m = 0; m < 24; m++)
					for (int n = 0; n < 24; n++)
					{
						//Add a bias
						conva_z[i][m][n] = conva_bias[i];
						for (int a = 0 ; a < 6; a++) //Point-wise Convolution
							conva_z[i][m][n] = conva_z[i][m][n] + conv1_z[a][m][n] * conva_weight[i][a];
					}

			//Add up Conv Layer and Serial Adapter Layer and do Activiation
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 24; m++)
					for (int n = 0; n < 24; n++)
					{
						conv_z[i][m][n] = conv1_z[i][m][n] + conva_z[i][m][n];
						conv_actv[i][m][n] = relu(conv_z[i][m][n]); //sigmoid(conv_z[i][m][n]);
					}
		}
		//Parallel Adapter
		else if (adapt == 2) {
			for (int i = 0; i < 6; i++) //To get six out channels
				for (int m = 0; m < 28; m++)
					for (int n = 0; n < 28; n++)
					{
						convpa_z[i][m][n] = convpa_bias[i];
						convpa_z[i][m][n] = convpa_z[i][m][n] + image[m][n] * convpa_weight[i]; 
					}

			//Add up Conv Layer and Serial Adapter Layer and do Activiation
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 28; m++)
					for (int n = 0; n < 28; n++)
					{
						conv_z_pa[i][m][n] = conv1_z_pa[i][m][n] + convpa_z[i][m][n];
						conv_actv_pa[i][m][n] = relu(conv_z_pa[i][m][n]); //sigmoid(conv_z_pa[i][m][n]);
					}
		}

		//Maxpooling Layer
		//Parallel Adapter
		if (adapt == 2) {
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 14; m++)
					for (int n = 0; n < 14; n++)
					{
						maxpool(conv_actv_pa[i][2 * m][2 * n], conv_actv_pa[i][2 * m][2 * n + 1], conv_actv_pa[i][2 * m + 1][2 * n], conv_actv_pa[i][2 * m + 1][2 * n + 1], &maxpool_z_pa[i][m][n], &pool_index_pa[i][m][n]);
						maxpool_actv_pa[i][m][n] = maxpool_z_pa[i][m][n];
					}
		}
		else {
			for (int i = 0; i < 6; i++)
				for (int m = 0; m < 12; m++)
					for (int n = 0; n < 12; n++)
					{
						maxpool(conv_actv[i][2 * m][2 * n], conv_actv[i][2 * m][2 * n + 1], conv_actv[i][2 * m + 1][2 * n], conv_actv[i][2 * m + 1][2 * n + 1], &maxpool_z[i][m][n], &pool_index[i][m][n]);
						maxpool_actv[i][m][n] = maxpool_z[i][m][n];
					}
		}

		//FC Hidden Layer
		if (adapt == 2) {
			for (int k = 0; k < 45; k++)
			{
				hidden_z[k] = 0.0;
				for (int i = 0; i < 6; i++)
				for (int m = 0; m < 14; m++)
				for (int n = 0; n < 14; n++)
				hidden_z[k] = hidden_z[k] + maxpool_actv_pa[i][m][n] * max_to_hidden_weights_pa[i][m][n][k];
				hidden_actv[k] = relu(hidden_z[k] + hidden_bias[k]); //sigmoid(hidden_z[k] + hidden_bias[k]);
			}
		}
		else {
			for (int k = 0; k < 45; k++)
			{
				hidden_z[k] = 0.0;
				for (int i = 0; i < 6; i++)
				for (int m = 0; m < 12; m++)
				for (int n = 0; n < 12; n++)
				hidden_z[k] = hidden_z[k] + maxpool_actv[i][m][n] * max_to_hidden_weights[i][m][n][k];
				hidden_actv[k] = relu(hidden_z[k] + hidden_bias[k]); //sigmoid(hidden_z[k] + hidden_bias[k]);
			}
		}

		//FC Output Layer
		if (network == 0)
		{
			for (int i = 0; i < 9; i++)
			{
				out_z1to9[i] = 0.0;
				for (int j = 0; j < 45; j++)
					out_z1to9[i] = out_z1to9[i] + hidden_actv[j] * hidden_to_out_weights1to9[j][i];

				out_actv1to9[i] = sigmoid(out_z1to9[i] + out_bias1to9[i]);
				outError1to9[i] = out_actv1to9[i] - actual_output1to9[i]; //Fundamental Equation 1 --> Output error = (aL - y) 
			}
		}
		else
		{
			for (int i = 0; i < 10; i++)
			{
				out_z[i] = 0.0;
				for (int j = 0; j < 45; j++)
					out_z[i] = out_z[i] + hidden_actv[j] * hidden_to_out_weights[j][i];

				out_actv[i] = sigmoid(out_z[i] + out_bias[i]);
				outError[i] = out_actv[i] - actual_output[i]; //Fundamental Equation 1 --> Output error = (aL - y) 
			}
		}
		/////////////////////////Forward Propoagation///////////////////////////////////

		if(network == 0)
		{
			if ( (max_out1to9(out_actv1to9) + 1) != label)
				count_test++;
		}
		else {
			if (max_out(out_actv) != label)
				count_test++;
		}

		////////////////////////////////Display progress///////////////////////////////////
		double progress = ((double)k + 1) / ((double)network_size) * 100.0;
	    printf("TESTING: reading image %5d / %5d progress [%3d%%]  ", k + 1, network_size, (int)progress);

	    float accuracy = (1.0 - ((float)count_test / (float)(k + 1))) * 100.0;
	    printf("TOTAL: correct=%5d  incorrect=%5d  accuracy=%5.4f%%  ", k + 1 - count_test, count_test, accuracy);


		if(network == 0)
		{
		    if ((max_out1to9(out_actv1to9)+1) != label) {
		        printf("PREDICTED: %1d ACTUAL: %1d\n", max_out1to9(out_actv1to9)+1, (int)label);
		    } else {
		        printf("PREDICTED: %1d ACTUAL: %1d\n", max_out1to9(out_actv1to9)+1, (int)label);
		    }
		}
		else
		{
			if (max_out(out_actv) != label) {
		        printf("PREDICTED: %1d ACTUAL: %1d\n", max_out(out_actv), (int)label);
		    } else {
		        printf("PREDICTED: %1d ACTUAL: %1d\n", max_out(out_actv), (int)label);
		    }
		}
	   ////////////////////////////////Display progress///////////////////////////////////
	}
}


void rewind(FILE *f);

float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}
float sigmoid_deri(float y)
{
	return y * (1 - y);
}
float relu(float x) 
{
	return (x > 0) ? x : 0;
}
float relu_deri(float y)
{
	return (y > 0) ? 1 : 0;
}
void maxpool(float a, float b, float c, float d, float *max_num, int *max_index)
{

	*max_num = a;
	*max_index = 0;
	if (b > *max_num)
	{
	*max_num = b;
	*max_index = 1;
	}
	if (c > *max_num)
	{
	*max_num = c;
	*max_index = 2;
	}
	if (d > *max_num)
	{
	*max_num = d;
	*max_index = 3;
	}
}

int max_out(float a[10])
{
	float max = 0.0;
	int index = 0;

	for (int c = 0; c < 10; c++)
	{
		if (a[c] > max)
		{
			index = c;
			max = a[c];
		}
	}

	return index;
}

int max_out1to9(float a[9])
{
	float max = 0.0;
	int index = 0;

	for (int c = 0; c < 9; c++)
	{
		if (a[c] > max)
		{
			index = c;
			max = a[c];
		}
	}

	return index;
}

void print_conv_wxb(int adapt)
{	
	printf("\nFor ADAPT %d Printing CONV Parameters\n", adapt);
	printf("\nConv Kernel Biases:\n");
	for (int i = 0; i < 6; i++)
		printf("%f ", conv_k_bias[i]);
	printf("\n");
	printf("\nConv Kernel Weights:\n");
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			for (int k = 0; k < 5; k++)
				printf("%f ", conv_k_weights[i][j][k]);
			printf("\n");
		}
		printf("\n");
		fflush(stdout);
	}
}

void print_convSA_wxb(int adapt)
{	
	printf("\nFor ADAPT %d Printing CONV Parameters\n", adapt);
	printf("\nConvSA Kernel Biases:\n");
	for (int i = 0; i < 6; i++)
		printf("%f ", conva_bias[i]);
	printf("\n");
	printf("\nConvSA Kernel Weights:\n");
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			printf("%f ", conva_weight[i][j]);
		}
		printf("\n");
		fflush(stdout);
	}
}

// Print for RTL

// MNIST
// Print SA / PA weights and biases

void print_MNIST_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int k_val)
{
	// float image[28][28]
	int length = dim1*dim2;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{

				float_union.float32 = image[ind_i][ind_j];
				floatBytes = (unsigned char *) & float_union.bytes;

				count = count + 1;

				/// Ending case no newline
				if(count >= length) {		
					// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, image[ind_i][ind_j]);
					// fprintf(file_ptr, "%f ", image[ind_i][ind_j]);
					for(int ind_o = (4-1); ind_o >= 0; ind_o--)
					{
						fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
					}
					
					
					if(newline == 1) {
						// fprintf(file_ptr, "   %d",count);
						fprintf(file_ptr, "\n");

					}	
				} 

				/// Save each value on a new line
				else 
				{

					// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, image[ind_i][ind_j]);
					// fprintf(file_ptr, "%f ", image[ind_i][ind_j]);
					for(int ind_o = (4-1); ind_o >= 0; ind_o--)
					{
						fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
					}
					fprintf(file_ptr, "\n");
				}	
				
			}
		}

		fclose(file_ptr);

	}	
}

void print_MNIST_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int k_val)
{
	// float image[28][28]
	int length = dim1*dim2;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{

				count = count + 1;

				/// Ending case no newline
				if(count >= length) {		
					// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, image[ind_i][ind_j]);		
					fprintf(file_ptr, "%f", image[ind_i][ind_j]);						
					
					if(newline == 1) {
						fprintf(file_ptr, "\n");
					}	
				} 

				/// Save each value on a new line
				else 
				{
					// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, image[ind_i][ind_j]);
					fprintf(file_ptr, "%f", image[ind_i][ind_j]);
					fprintf(file_ptr, "\n");
				}	
				
			}
		}

		fclose(file_ptr);

	}	
}


// Conv_z eg fmap sum
void print_SA_sum_fmap_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float conv_z[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{
					float_union.float32 = conv_z[ind_i][ind_j][ind_k];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv_z[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", conv_z[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv_z[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", conv_z[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}


void print_SA_sum_fmap_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{

	// float conv_z[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv_z[ind_i][ind_j][ind_k]);		
						fprintf(file_ptr, "%f", conv_z[ind_i][ind_j][ind_k]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv_z[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "%f", conv_z[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}


void print_PA_sum_fmap_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{

	// float conv_z_pa[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{
					float_union.float32 = conv_z_pa[ind_i][ind_j][ind_k];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv_z_pa[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", conv_z_pa[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv_z_pa[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", conv_z_pa[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

void print_PA_sum_fmap_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float conv_z_pa[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv_z_pa[ind_i][ind_j][ind_k]);		
						fprintf(file_ptr, "%f", conv_z_pa[ind_i][ind_j][ind_k]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}	
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv_z_pa[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "%f", conv_z_pa[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}



// Conv1_z
void print_SA_fmap_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{

	// float conv1_z[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{
					float_union.float32 = conv1_z[ind_i][ind_j][ind_k];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", conv1_z[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", conv1_z[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

void print_SA_fmap_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float conv1_z[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z[ind_i][ind_j][ind_k]);		
						fprintf(file_ptr, "%f", conv1_z[ind_i][ind_j][ind_k]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}	
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "%f", conv1_z[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}


void print_PA_fmap_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float conv1_z_pa[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{
					float_union.float32 = conv1_z_pa[ind_i][ind_j][ind_k];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z_pa[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", conv1_z_pa[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z_pa[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", conv1_z_pa[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

void print_PA_fmap_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float conv1_z_pa[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z_pa[ind_i][ind_j][ind_k]);		
						fprintf(file_ptr, "%f", conv1_z_pa[ind_i][ind_j][ind_k]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}	
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z_pa[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "%f", conv1_z_pa[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}



// Conv1_z by channels
void print_SA_fmap_bychan_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{

	// float conv1_z[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim2; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim3; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim1; ind_k++) 
				{
					float_union.float32 = conv1_z[ind_k][ind_i][ind_j];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z[ind_k][ind_i][ind_j]);
						// fprintf(file_ptr, "%f ", conv1_z[ind_k][ind_i][ind_j]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k,conv1_z[ind_k][ind_i][ind_j]);
						// fprintf(file_ptr, "%f ", conv1_z[ind_k][ind_i][ind_j]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

void print_SA_fmap_bychan_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float conv1_z[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim2; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim3; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim1; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z[ind_k][ind_i][ind_j]);		
						fprintf(file_ptr, "%f", conv1_z[ind_k][ind_i][ind_j]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}	
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z[ind_k][ind_i][ind_j]);
						fprintf(file_ptr, "%f", conv1_z[ind_k][ind_i][ind_j]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}


void print_PA_fmap_bychan_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float conv1_z_pa[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim2; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim3; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim1; ind_k++) 
				{
					float_union.float32 = conv1_z_pa[ind_k][ind_i][ind_j];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z_pa[ind_k][ind_i][ind_j]);
						// fprintf(file_ptr, "%f ", conv1_z_pa[ind_k][ind_i][ind_j]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z_pa[ind_k][ind_i][ind_j]);
						// fprintf(file_ptr, "%f ", conv1_z_pa[ind_k][ind_i][ind_j]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

void print_PA_fmap_bychan_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float conv1_z_pa[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim2; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim3; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim1; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z_pa[ind_k][ind_i][ind_j]);		
						fprintf(file_ptr, "%f", conv1_z_pa[ind_k][ind_i][ind_j]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}	
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conv1_z_pa[ind_k][ind_i][ind_j]);
						fprintf(file_ptr, "%f", conv1_z_pa[ind_k][ind_i][ind_j]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}




// Adapter z
void print_SA_z_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float conva_z[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{
					float_union.float32 = conva_z[ind_i][ind_j][ind_k];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_z[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", conva_z[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_z[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", conva_z[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

// float conva_z[6][24][24]
void print_SA_z_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float conva_z[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_z[ind_i][ind_j][ind_k]);		
						fprintf(file_ptr, "%f", conva_z[ind_i][ind_j][ind_k]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}	
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_z[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "%f", conva_z[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}


void print_PA_z_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float convpa_z[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{
					float_union.float32 = convpa_z[ind_i][ind_j][ind_k];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_z[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", convpa_z[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_z[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", convpa_z[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

void print_PA_z_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float convpa_z[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_z[ind_i][ind_j][ind_k]);		
						fprintf(file_ptr, "%f", convpa_z[ind_i][ind_j][ind_k]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}	
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_z[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "%f", convpa_z[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}




// Print SA / PA weights and biases

void print_SA_weights_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int k_val)
{
	// float conva_weight[6][6]
	int length = dim1*dim2;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{

				float_union.float32 = conva_weight[ind_i][ind_j];
				floatBytes = (unsigned char *) &float_union.bytes;

				count = count + 1;

				/// Ending case no newline
				if(count >= length) {		
					// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_weight[ind_i][ind_j]);
					// fprintf(file_ptr, "%f ", conva_weight[ind_i][ind_j]);
					for(int ind_o = (4-1); ind_o >= 0; ind_o--)
					{
						fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
					}
					
					
					if(newline == 1) {
						// fprintf(file_ptr, "   %d",count);
						fprintf(file_ptr, "\n");

					}	
				} 

				/// Save each value on a new line
				else 
				{

					// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_weight[ind_i][ind_j]);
					// fprintf(file_ptr, "%f ", conva_weight[ind_i][ind_j]);
					for(int ind_o = (4-1); ind_o >= 0; ind_o--)
					{
						fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
					}
					fprintf(file_ptr, "\n");
				}	
				
			}
		}

		fclose(file_ptr);

	}	
}

void print_SA_weights_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int k_val)
{
	// float conva_weight[6][6]
	int length = dim1*dim2;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{

				count = count + 1;

				/// Ending case no newline
				if(count >= length) {		
					// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_weight[ind_i][ind_j]);		
					fprintf(file_ptr, "%f", conva_weight[ind_i][ind_j]);						
					
					if(newline == 1) {
						fprintf(file_ptr, "\n");
					}	
				} 

				/// Save each value on a new line
				else 
				{
					// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_weight[ind_i][ind_j]);
					fprintf(file_ptr, "%f", conva_weight[ind_i][ind_j]);
					fprintf(file_ptr, "\n");
				}	
				
			}
		}

		fclose(file_ptr);

	}	
}

void print_SA_biases_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val)
{
	// float conva_bias[6]
	int length = dim1;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{

			float_union.float32 = conva_bias[ind_i];
			floatBytes = (unsigned char *) &float_union.bytes;

			count = count + 1;

			/// Ending case no newline
			if(count >= length) {		
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_bias[ind_i]);
				// fprintf(file_ptr, "%f ", conva_bias[ind_i]);
				for(int ind_o = (4-1); ind_o >= 0; ind_o--)
				{
					fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
				}
				
				
				if(newline == 1) {
					// fprintf(file_ptr, "   %d",count);
					fprintf(file_ptr, "\n");

				}	
			} 

			/// Save each value on a new line
			else 
			{

				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_bias[ind_i]);
				// fprintf(file_ptr, "%f ", conva_bias[ind_i]);
				for(int ind_o = (4-1); ind_o >= 0; ind_o--)
				{
					fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
				}
				fprintf(file_ptr, "\n");
			}
				
			
		}

		fclose(file_ptr);

	}	
}

void print_SA_biases_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val)
{
	// float conva_bias[6]
	int length = dim1;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{

			count = count + 1;

			/// Ending case no newline
			if(count >= length) {		
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_bias[ind_i]);		
				fprintf(file_ptr, "%f", conva_bias[ind_i]);						
				
				if(newline == 1) {
					fprintf(file_ptr, "\n");
				}	
			} 

			/// Save each value on a new line
			else 
			{
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_bias[ind_i]);
				fprintf(file_ptr, "%f", conva_bias[ind_i]);
				fprintf(file_ptr, "\n");
			}	
				
			
		}

		fclose(file_ptr);

	}	
}



void print_PA_weights_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val)
{

	// float convpa_weight[6]
	int length = dim1;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{

			float_union.float32 = convpa_weight[ind_i];
			floatBytes = (unsigned char *) &float_union.bytes;

			count = count + 1;

			/// Ending case no newline
			if(count >= length) {		
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_weight[ind_i]);
				// fprintf(file_ptr, "%f ", convpa_weight[ind_i]);
				for(int ind_o = (4-1); ind_o >= 0; ind_o--)
				{
					fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
				}
				
				
				if(newline == 1) {
					// fprintf(file_ptr, "   %d",count);
					fprintf(file_ptr, "\n");

				}	
			} 

			/// Save each value on a new line
			else 
			{

				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_weight[ind_i]);
				// fprintf(file_ptr, "%f ", convpa_weight[ind_i]);
				for(int ind_o = (4-1); ind_o >= 0; ind_o--)
				{
					fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
				}
				fprintf(file_ptr, "\n");
			}
				
			
		}

		fclose(file_ptr);

	}	
}

void print_PA_weights_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val)
{
	// float convpa_weight[6]
	int length = dim1;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{

			count = count + 1;

			/// Ending case no newline
			if(count >= length) {		
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_weight[ind_i]);		
				fprintf(file_ptr, "%f", convpa_weight[ind_i]);						
				
				if(newline == 1) {
					fprintf(file_ptr, "\n");
				}	
			} 

			/// Save each value on a new line
			else 
			{
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_weight[ind_i]);
				fprintf(file_ptr, "%f", convpa_weight[ind_i]);
				fprintf(file_ptr, "\n");
			}	
				
			
		}

		fclose(file_ptr);

	}	
}


void print_PA_biases_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val)
{
	// float convpa_bias[6]
	int length = dim1;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{

			float_union.float32 = convpa_bias[ind_i];
			floatBytes = (unsigned char *) &float_union.bytes;

			count = count + 1;

			/// Ending case no newline
			if(count >= length) {		
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_bias[ind_i]);
				// fprintf(file_ptr, "%f ", convpa_bias[ind_i]);
				for(int ind_o = (4-1); ind_o >= 0; ind_o--)
				{
					fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
				}
				
				
				if(newline == 1) {
					// fprintf(file_ptr, "   %d",count);
					fprintf(file_ptr, "\n");

				}	
			} 

			/// Save each value on a new line
			else 
			{

				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_bias[ind_i]);
				// fprintf(file_ptr, "%f ", convpa_bias[ind_i]);
				for(int ind_o = (4-1); ind_o >= 0; ind_o--)
				{
					fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
				}
				fprintf(file_ptr, "\n");
			}
				
			
		}

		fclose(file_ptr);

	}	
}

void print_PA_biases_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val)
{
	// float convpa_bias[6]
	int length = dim1;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{

			count = count + 1;

			/// Ending case no newline
			if(count >= length) {		
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_bias[ind_i]);		
				fprintf(file_ptr, "%f", convpa_bias[ind_i]);						
				
				if(newline == 1) {
					fprintf(file_ptr, "\n");
				}	
			} 

			/// Save each value on a new line
			else 
			{
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_bias[ind_i]);
				fprintf(file_ptr, "%f", convpa_bias[ind_i]);
				fprintf(file_ptr, "\n");
			}	
				
			
		}

		fclose(file_ptr);

	}	
}



// Print Delta SA / PA weights and biases

void print_SA_delta_weights_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int k_val)
{
	// float conva_dw[6][6]
	int length = dim1*dim2;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{

				float_union.float32 = conva_dw[ind_i][ind_j];
				floatBytes = (unsigned char *) &float_union.bytes;

				count = count + 1;

				/// Ending case no newline
				if(count >= length) {		
					// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_dw[ind_i][ind_j]);
					// fprintf(file_ptr, "%f ", conva_dw[ind_i][ind_j]);
					for(int ind_o = (4-1); ind_o >= 0; ind_o--)
					{
						fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
					}
					
					
					if(newline == 1) {
						// fprintf(file_ptr, "   %d",count);
						fprintf(file_ptr, "\n");

					}	
				} 

				/// Save each value on a new line
				else 
				{

					// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_dw[ind_i][ind_j]);
					// fprintf(file_ptr, "%f ", conva_dw[ind_i][ind_j]);
					for(int ind_o = (4-1); ind_o >= 0; ind_o--)
					{
						fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
					}
					fprintf(file_ptr, "\n");
				}	
				
			}
		}

		fclose(file_ptr);

	}	
}

void print_SA_delta_weights_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int k_val)
{

	// float conva_dw[6][6]
	int length = dim1*dim2;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{

				count = count + 1;

				/// Ending case no newline
				if(count >= length) {		
					// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_dw[ind_i][ind_j]);		
					fprintf(file_ptr, "%f", conva_dw[ind_i][ind_j]);						
					
					if(newline == 1) {
						fprintf(file_ptr, "\n");
					}	
				} 

				/// Save each value on a new line
				else 
				{
					// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_dw[ind_i][ind_j]);
					fprintf(file_ptr, "%f", conva_dw[ind_i][ind_j]);
					fprintf(file_ptr, "\n");
				}	
				
			}
		}

		fclose(file_ptr);

	}	
}

void print_SA_delta_biases_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val)
{
	// float conva_db[6]
	int length = dim1;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{

			float_union.float32 = conva_db[ind_i];
			floatBytes = (unsigned char *) &float_union.bytes;

			count = count + 1;

			/// Ending case no newline
			if(count >= length) {		
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_db[ind_i]);
				// fprintf(file_ptr, "%f ", conva_db[ind_i]);
				for(int ind_o = (4-1); ind_o >= 0; ind_o--)
				{
					fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
				}
				
				
				if(newline == 1) {
					// fprintf(file_ptr, "   %d",count);
					fprintf(file_ptr, "\n");

				}	
			} 

			/// Save each value on a new line
			else 
			{

				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_db[ind_i]);
				// fprintf(file_ptr, "%f ", conva_db[ind_i]);
				for(int ind_o = (4-1); ind_o >= 0; ind_o--)
				{
					fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
				}
				fprintf(file_ptr, "\n");
			}
				
			
		}

		fclose(file_ptr);

	}	
}

void print_SA_delta_biases_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val)
{

	// float conva_db[6]
	int length = dim1;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{

			count = count + 1;

			/// Ending case no newline
			if(count >= length) {		
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_db[ind_i]);		
				fprintf(file_ptr, "%f", conva_db[ind_i]);						
				
				if(newline == 1) {
					fprintf(file_ptr, "\n");
				}	
			} 

			/// Save each value on a new line
			else 
			{
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, conva_db[ind_i]);
				fprintf(file_ptr, "%f", conva_db[ind_i]);
				fprintf(file_ptr, "\n");
			}	
				
			
		}

		fclose(file_ptr);

	}	
}



void print_PA_delta_weights_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val)
{

	// float convpa_dw[6]
	int length = dim1;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{

			float_union.float32 = convpa_dw[ind_i];
			floatBytes = (unsigned char *) &float_union.bytes;

			count = count + 1;

			/// Ending case no newline
			if(count >= length) {		
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_dw[ind_i]);
				// fprintf(file_ptr, "%f ", convpa_dw[ind_i]);
				for(int ind_o = (4-1); ind_o >= 0; ind_o--)
				{
					fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
				}
				
				
				if(newline == 1) {
					// fprintf(file_ptr, "   %d",count);
					fprintf(file_ptr, "\n");

				}	
			} 

			/// Save each value on a new line
			else 
			{

				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_dw[ind_i]);
				// fprintf(file_ptr, "%f ", convpa_dw[ind_i]);
				for(int ind_o = (4-1); ind_o >= 0; ind_o--)
				{
					fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
				}
				fprintf(file_ptr, "\n");
			}
				
			
		}

		fclose(file_ptr);

	}	
}

void print_PA_delta_weights_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val)
{

	// float convpa_dw[6]
	int length = dim1;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{

			count = count + 1;

			/// Ending case no newline
			if(count >= length) {		
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_dw[ind_i]);		
				fprintf(file_ptr, "%f", convpa_dw[ind_i]);						
				
				if(newline == 1) {
					fprintf(file_ptr, "\n");
				}	
			} 

			/// Save each value on a new line
			else 
			{
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_dw[ind_i]);
				fprintf(file_ptr, "%f", convpa_dw[ind_i]);
				fprintf(file_ptr, "\n");
			}	
				
			
		}

		fclose(file_ptr);

	}	
}


void print_PA_delta_biases_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val)
{
	// float convpa_db[6]
	int length = dim1;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{

			float_union.float32 = convpa_db[ind_i];
			floatBytes = (unsigned char *) &float_union.bytes;

			count = count + 1;

			/// Ending case no newline
			if(count >= length) {		
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_db[ind_i]);
				// fprintf(file_ptr, "%f ", convpa_db[ind_i]);
				for(int ind_o = (4-1); ind_o >= 0; ind_o--)
				{
					fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
				}
				
				
				if(newline == 1) {
					// fprintf(file_ptr, "   %d",count);
					fprintf(file_ptr, "\n");

				}	
			} 

			/// Save each value on a new line
			else 
			{

				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_db[ind_i]);
				// fprintf(file_ptr, "%f ", convpa_db[ind_i]);
				for(int ind_o = (4-1); ind_o >= 0; ind_o--)
				{
					fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
				}
				fprintf(file_ptr, "\n");
			}
				
			
		}

		fclose(file_ptr);

	}	
}

void print_PA_delta_biases_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int k_val)
{

	// float convpa_db[6]
	int length = dim1;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{

			count = count + 1;

			/// Ending case no newline
			if(count >= length) {		
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_db[ind_i]);		
				fprintf(file_ptr, "%f", convpa_db[ind_i]);						
				
				if(newline == 1) {
					fprintf(file_ptr, "\n");
				}	
			} 

			/// Save each value on a new line
			else 
			{
				// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, convpa_db[ind_i]);
				fprintf(file_ptr, "%f", convpa_db[ind_i]);
				fprintf(file_ptr, "\n");
			}	
				
			
		}

		fclose(file_ptr);

	}	
}


// Max pooling

// float maxpool_error[6][24][24]
void print_SA_maxpool_error_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float maxpool_error[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{
					float_union.float32 = maxpool_error[ind_i][ind_j][ind_k];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, maxpool_error[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", maxpool_error[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, maxpool_error[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", maxpool_error[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

// float maxpool_error[6][24][24]
void print_SA_maxpool_error_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float maxpool_error[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, maxpool_error[ind_i][ind_j][ind_k]);		
						fprintf(file_ptr, "%f", maxpool_error[ind_i][ind_j][ind_k]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}	
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, maxpool_error[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "%f", maxpool_error[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

// float maxpool_error_pa[6][28][28]
void print_PA_maxpool_error_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float maxpool_error_pa[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{
					float_union.float32 = maxpool_error_pa[ind_i][ind_j][ind_k];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, maxpool_error_pa[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", maxpool_error_pa[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, maxpool_error_pa[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", maxpool_error_pa[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

// float maxpool_error_pa[6][28][28]
void print_PA_maxpool_error_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float maxpool_error_pa[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, maxpool_error_pa[ind_i][ind_j][ind_k]);		
						fprintf(file_ptr, "%f", maxpool_error_pa[ind_i][ind_j][ind_k]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}	
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, maxpool_error_pa[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "%f", maxpool_error_pa[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}



// Deriv_actv
// float deriv_actv[6][24][24]
void print_SA_deriv_actv_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float deriv_actv[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{
					float_union.float32 = deriv_actv[ind_i][ind_j][ind_k];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, deriv_actv[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", deriv_actv[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, deriv_actv[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", deriv_actv[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

// float deriv_actv[6][24][24]
void print_SA_deriv_actv_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float deriv_actv[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, deriv_actv[ind_i][ind_j][ind_k]);		
						fprintf(file_ptr, "%f", deriv_actv[ind_i][ind_j][ind_k]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}	
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, deriv_actv[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "%f", deriv_actv[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

// float deriv_actv_pa[6][28][28]
void print_PA_deriv_actv_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float deriv_actv_pa[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{
					float_union.float32 = deriv_actv_pa[ind_i][ind_j][ind_k];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, deriv_actv_pa[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", deriv_actv_pa[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, deriv_actv_pa[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", deriv_actv_pa[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

// float deriv_actv_pa[6][28][28]
void print_PA_deriv_actv_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float deriv_actv_pa[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, deriv_actv_pa[ind_i][ind_j][ind_k]);		
						fprintf(file_ptr, "%f", deriv_actv_pa[ind_i][ind_j][ind_k]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}	
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, deriv_actv_pa[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "%f", deriv_actv_pa[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}


// delta_error
// float hidden1ErrorSum[6][24][24]
void print_SA_delta_error_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float hidden1ErrorSum[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{
					float_union.float32 = hidden1ErrorSum[ind_i][ind_j][ind_k];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, hidden1ErrorSum[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", hidden1ErrorSum[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, hidden1ErrorSum[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", hidden1ErrorSum[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

// float hidden1ErrorSum[6][24][24]
void print_SA_delta_error_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float hidden1ErrorSum[6][24][24]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, hidden1ErrorSum[ind_i][ind_j][ind_k]);		
						fprintf(file_ptr, "%f", hidden1ErrorSum[ind_i][ind_j][ind_k]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}	
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, hidden1ErrorSum[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "%f", hidden1ErrorSum[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

// float hidden1ErrorSum_pa[6][28][28]
void print_PA_delta_error_hex(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float hidden1ErrorSum_pa[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		unsigned char * floatBytes = NULL;
		union Float32Union float_union;

		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{
					float_union.float32 = hidden1ErrorSum_pa[ind_i][ind_j][ind_k];
					floatBytes = (unsigned char *) &float_union.bytes;

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, hidden1ErrorSum_pa[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", hidden1ErrorSum_pa[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						
						
						if(newline == 1) {
							// fprintf(file_ptr, "   %d",count);
							fprintf(file_ptr, "\n");

						}	
					} 

					/// Save each value on a new line
					else 
					{

						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, hidden1ErrorSum_pa[ind_i][ind_j][ind_k]);
						// fprintf(file_ptr, "%f ", hidden1ErrorSum_pa[ind_i][ind_j][ind_k]);
						for(int ind_o = (4-1); ind_o >= 0; ind_o--)
						{
							fprintf(file_ptr, "%02X", floatBytes[ind_o]);					
						}
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}

// float hidden1ErrorSum_pa[6][28][28]
void print_PA_delta_error_float(int adapt, const char * name, int print_to_file, const char * path, const char * mode, int newline, int precision, int dim1, int dim2, int dim3, int k_val)
{
	// float hidden1ErrorSum_pa[6][28][28]
	int length = dim1*dim2*dim3;
	int count = 0;
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;

		// char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int ind_i = 0; ind_i < dim1; ind_i++) 
		{
			for(int ind_j = 0; ind_j < dim2; ind_j++) 
			{
				for(int ind_k = 0; ind_k < dim3; ind_k++) 
				{

					count = count + 1;

					/// Ending case no newline
					if(count >= length) {		
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, hidden1ErrorSum_pa[ind_i][ind_j][ind_k]);		
						fprintf(file_ptr, "%f", hidden1ErrorSum_pa[ind_i][ind_j][ind_k]);						
						
						if(newline == 1) {
							fprintf(file_ptr, "\n");
						}	
					} 

					/// Save each value on a new line
					else 
					{
						// fprintf(file_ptr, "%d %d %d %d %f ", k_val,ind_i,ind_j,ind_k, hidden1ErrorSum_pa[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "%f", hidden1ErrorSum_pa[ind_i][ind_j][ind_k]);
						fprintf(file_ptr, "\n");
					}	
				}
			}
		}

		fclose(file_ptr);

	}	
}








void print_conv_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1, int dim2, int dim3)
{	
	int length = dim1*dim2*dim3;
	int count = 0;

	// float conv_k_weights[6][5][5], 

	printf("\nPrinting %s : ADAPT %d : Print %d : PrinttoFile %d : Path %s : Mode %s\n\n%s\n", name, adapt, print_to_stdout, print_to_file, path, mode, name);

	if(print_to_stdout == 1) 
	{
		count = 0;
		for(int i = 0; i < dim1; i++) 
		{
			for(int j = 0; j < dim2; j++) 
			{
				printf("\n%d/%d %d/%d\n", i+1, dim1, j+1, dim2);
				for(int k = 0; k < dim3; k++) 
				{
					count = count + 1;
					printf("CONV1 Weights : Index %d/%d : %f\n", count, length, conv_k_weights[i][j][k]);	
				}
			}
		}
	}		
	printf("\n");
	fflush(stdout);
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;
		char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int i = 0; i < dim1; i++) 
		{
			for(int j = 0; j < dim2; j++) 
			{
				for(int k = 0; k < dim3; k++) 
				{
					count = count + 1;
					if(count >= length) {
						fprintf(file_ptr, "%s", gcvt(conv_k_weights[i][j][k], precision, float_buffer));	
					} 
					else 
					{
						fprintf(file_ptr, "%s\n", gcvt(conv_k_weights[i][j][k], precision, float_buffer));	
					}	
				}
			}
		}

		fclose(file_ptr);

	}
	
}

void print_conv_biases(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1)
{	
	// float conv_k_bias[6]
	int count = 0;
	int length = dim1;

	printf("\nPrinting %s : ADAPT %d : Print %d : PrinttoFile %d : Path %s : Mode %s\n\n%s\n",  name, adapt, print_to_stdout, print_to_file, path, mode, name);

	if(print_to_stdout == 1) 
	{
		for (int i = 0; i < dim1; i++)
		{
			printf("%d / %d : %f\n", i+1, dim1, conv_k_bias[i]);	
		}		
	printf("\n");
	fflush(stdout);
	}

	if(print_to_file == 1)
	{
		FILE * file_ptr = NULL;
		char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for (int i = 0; i < dim1; i++)
		{
			count = count + 1;
			if(count >= length) {

				fprintf(file_ptr, "  %s", gcvt(conv_k_bias[i], precision, float_buffer));
				
			}
			else
			{
				
				fprintf(file_ptr, "  %s", gcvt(conv_k_bias[i], precision, float_buffer));
				fprintf(file_ptr, "\n");				
			}
			
		}

		fclose(file_ptr);


	}
	
}


void print_SA_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1, int dim2)
{	
	int length = dim1*dim2;
	int count = 0;

	// float conva_weight[6][6];

	printf("\nPrinting %s : ADAPT %d : Print %d : PrinttoFile %d : Path %s : Mode %s\n\n%s\n", name, adapt, print_to_stdout, print_to_file, path, mode, name);

	if(print_to_stdout == 1) 
	{
		count = 0;
		for(int i = 0; i < dim1; i++) 
		{
			printf("\n%d/%d\n", i+1, dim1);
			for(int j = 0; j < dim2; j++) 
			{				
				count = count + 1;
				printf("SA Weights : Index %d/%d : %f\n", count, length, conva_weight[i][j]);					
			}
		}
	}		
	printf("\n");
	fflush(stdout);
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;
		char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int i = 0; i < dim1; i++) 
		{
			for(int j = 0; j < dim2; j++) 
			{
				count = count + 1;
				if(count >= length) {
					fprintf(file_ptr, "%s", gcvt(conva_weight[i][j], precision, float_buffer));	
				} 
				else 
				{
					fprintf(file_ptr, "%s\n", gcvt(conva_weight[i][j], precision, float_buffer));	
				}	
				
			}
		}

		fclose(file_ptr);

	}
	
}

void print_SA_biases(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1)
{	
	// float conva_bias[6];
	int count = 0;
	int length = dim1;

	printf("\nPrinting %s : ADAPT %d : Print %d : PrinttoFile %d : Path %s : Mode %s\n\n%s\n",  name, adapt, print_to_stdout, print_to_file, path, mode, name);

	if(print_to_stdout == 1) 
	{
		for (int i = 0; i < dim1; i++)
		{
			printf("%d / %d : %f\n", i+1, dim1, conva_bias[i]);	
		}		
	printf("\n");
	fflush(stdout);
	}

	if(print_to_file == 1)
	{
		FILE * file_ptr = NULL;
		char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for (int i = 0; i < dim1; i++)
		{
			count = count + 1;
			if(count >= length) {
				fprintf(file_ptr, "%s", gcvt(conva_bias[i], precision, float_buffer));
			}
			else
			{
				fprintf(file_ptr, "%s\n", gcvt(conva_bias[i], precision, float_buffer));
			}
			
		}
		fclose(file_ptr);
	}
}

void print_PA_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1)
{	
	// float convpa_weight[6];
	int count = 0;
	int length = dim1;

	printf("\nPrinting %s : ADAPT %d : Print %d : PrinttoFile %d : Path %s : Mode %s\n\n%s\n",  name, adapt, print_to_stdout, print_to_file, path, mode, name);

	if(print_to_stdout == 1) 
	{
		for (int i = 0; i < dim1; i++)
		{
			printf("%d / %d : %f\n", i+1, dim1, convpa_weight[i]);	
		}		
	printf("\n");
	fflush(stdout);
	}

	if(print_to_file == 1)
	{
		FILE * file_ptr = NULL;
		char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for (int i = 0; i < dim1; i++)
		{
			count = count + 1;
			if(count >= length) {
				fprintf(file_ptr, "%s", gcvt(convpa_weight[i], precision, float_buffer));
			}
			else
			{
				fprintf(file_ptr, "%s\n", gcvt(convpa_weight[i], precision, float_buffer));
			}
			
		}
		fclose(file_ptr);
	}
}

void print_PA_biases(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1)
{	
	// float convpa_bias[6];
	int count = 0;
	int length = dim1;

	printf("\nPrinting %s : ADAPT %d : Print %d : PrinttoFile %d : Path %s : Mode %s\n\n%s\n",  name, adapt, print_to_stdout, print_to_file, path, mode, name);

	if(print_to_stdout == 1) 
	{
		for (int i = 0; i < dim1; i++)
		{
			printf("%d / %d : %f\n", i+1, dim1, convpa_bias[i]);	
		}		
	printf("\n");
	fflush(stdout);
	}

	if(print_to_file == 1)
	{
		FILE * file_ptr = NULL;
		char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for (int i = 0; i < dim1; i++)
		{
			count = count + 1;
			if(count >= length) {
				fprintf(file_ptr, "%s", gcvt(convpa_bias[i], precision, float_buffer));
			}
			else
			{
				fprintf(file_ptr, "%s\n", gcvt(convpa_bias[i], precision, float_buffer));
			}
			
		}
		fclose(file_ptr);
	}
}


void print_fc_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1, int dim2, int dim3, int dim4)
{	
	int length = dim1*dim2*dim3*dim4;
	int count = 0;

	// max_to_hidden_weights[6][12][12][45] FC WEIGHT

	printf("\nPrinting %s : ADAPT %d : Print %d : PrinttoFile %d : Path %s : Mode %s\n\n%s\n", name, adapt, print_to_stdout, print_to_file, path, mode, name);

	if(print_to_stdout == 1) 
	{
		count = 0;
		for(int i = 0; i < dim1; i++) 
		{
			for(int j = 0; j < dim2; j++) 
			{
				
				for(int k = 0; k < dim3; k++) 
				{
					printf("\n%d/%d %d/%d %d/%d\n", i+1, dim1, j+1, dim2, k+1, dim3);
					for(int l = 0; l < dim4; l++) 
					{
						count = count + 1;
						printf("FC Weights : Index %d/%d : %f\n", count, length, max_to_hidden_weights[i][j][k][l]);							
					}
				}
			}
		}
	}		
	printf("\n");
	fflush(stdout);
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;
		char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int i = 0; i < dim1; i++) 
		{
			for(int j = 0; j < dim2; j++) 
			{
				for(int k = 0; k < dim3; k++) 
				{
					for(int l = 0; l < dim4; l++) 
					{
						count = count + 1;

						if(count >= length) {
						fprintf(file_ptr, "%s", gcvt(max_to_hidden_weights[i][j][k][l], precision, float_buffer));	
						} 
						else 
						{
							fprintf(file_ptr, "%s\n", gcvt(max_to_hidden_weights[i][j][k][l], precision, float_buffer));	
						}								
					}					
				}
			}
		}

		fclose(file_ptr);

	}
	
}

void print_fcPA_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1, int dim2, int dim3, int dim4)
{	
	int length = dim1*dim2*dim3*dim4;
	int count = 0;

	// max_to_hidden_weights[6][12][12][45] FC WEIGHT

	printf("\nPrinting %s : ADAPT %d : Print %d : PrinttoFile %d : Path %s : Mode %s\n\n%s\n", name, adapt, print_to_stdout, print_to_file, path, mode, name);

	if(print_to_stdout == 1) 
	{
		count = 0;
		for(int i = 0; i < dim1; i++) 
		{
			for(int j = 0; j < dim2; j++) 
			{
				
				for(int k = 0; k < dim3; k++) 
				{
					printf("\n%d/%d %d/%d %d/%d\n", i+1, dim1, j+1, dim2, k+1, dim3);
					for(int l = 0; l < dim4; l++) 
					{
						count = count + 1;
						printf("fcPA Weights : Index %d/%d : %f\n", count, length, max_to_hidden_weights_pa[i][j][k][l]);							
					}
				}
			}
		}
	}		
	printf("\n");
	fflush(stdout);
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;
		char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int i = 0; i < dim1; i++) 
		{
			for(int j = 0; j < dim2; j++) 
			{
				for(int k = 0; k < dim3; k++) 
				{
					for(int l = 0; l < dim4; l++) 
					{
						count = count + 1;

						if(count >= length) {
						fprintf(file_ptr, "%s", gcvt(max_to_hidden_weights_pa[i][j][k][l], precision, float_buffer));	
						} 
						else 
						{
							fprintf(file_ptr, "%s\n", gcvt(max_to_hidden_weights_pa[i][j][k][l], precision, float_buffer));	
						}								
					}					
				}
			}
		}

		fclose(file_ptr);

	}
	
}

void print_fc_biases(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1)
{	
	// float convpa_bias[6];
	int count = 0;
	int length = dim1;

	printf("\nPrinting %s : ADAPT %d : Print %d : PrinttoFile %d : Path %s : Mode %s\n\n%s\n",  name, adapt, print_to_stdout, print_to_file, path, mode, name);

	if(print_to_stdout == 1) 
	{
		for (int i = 0; i < dim1; i++)
		{
			printf("%d / %d : %f\n", i+1, dim1, hidden_bias[i]);	
		}		
	printf("\n");
	fflush(stdout);
	}

	if(print_to_file == 1)
	{
		FILE * file_ptr = NULL;
		char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for (int i = 0; i < dim1; i++)
		{
			count = count + 1;
			if(count >= length) {
				fprintf(file_ptr, "%s", gcvt(hidden_bias[i], precision, float_buffer));
			}
			else
			{
				fprintf(file_ptr, "%s\n", gcvt(hidden_bias[i], precision, float_buffer));
			}
			
		}
		fclose(file_ptr);
	}
}

void print_out_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1, int dim2)
{	
	int length = dim1*dim2;
	int count = 0;

	// hidden_to_out_weights[45][10]

	printf("\nPrinting %s : ADAPT %d : Print %d : PrinttoFile %d : Path %s : Mode %s\n\n%s\n", name, adapt, print_to_stdout, print_to_file, path, mode, name);

	if(print_to_stdout == 1) 
	{
		count = 0;
		for(int i = 0; i < dim1; i++) 
		{
			printf("\n%d/%d\n", i+1, dim1);
			for(int j = 0; j < dim2; j++) 
			{				
				count = count + 1;
				printf("Out Weights : Index %d/%d : %f\n", count, length, hidden_to_out_weights[i][j]);					
			}
		}
	}		
	printf("\n");
	fflush(stdout);
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;
		char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int i = 0; i < dim1; i++) 
		{
			for(int j = 0; j < dim2; j++) 
			{
				count = count + 1;
				if(count >= length) {
					fprintf(file_ptr, "%s", gcvt(hidden_to_out_weights[i][j], precision, float_buffer));	
				} 
				else 
				{
					fprintf(file_ptr, "%s\n", gcvt(hidden_to_out_weights[i][j], precision, float_buffer));	
				}	
				
			}
		}

		fclose(file_ptr);

	}
	
}

void print_out_biases(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1)
{	
	// float convpa_bias[6];
	int count = 0;
	int length = dim1;

	printf("\nPrinting %s : ADAPT %d : Print %d : PrinttoFile %d : Path %s : Mode %s\n\n%s\n",  name, adapt, print_to_stdout, print_to_file, path, mode, name);

	if(print_to_stdout == 1) 
	{
		for (int i = 0; i < dim1; i++)
		{
			printf("%d / %d : %f\n", i+1, dim1, out_bias[i]);	
		}		
	printf("\n");
	fflush(stdout);
	}

	if(print_to_file == 1)
	{
		FILE * file_ptr = NULL;
		char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for (int i = 0; i < dim1; i++)
		{
			count = count + 1;
			if(count >= length) {
				fprintf(file_ptr, "%s", gcvt(out_bias[i], precision, float_buffer));
			}
			else
			{
				fprintf(file_ptr, "%s\n", gcvt(out_bias[i], precision, float_buffer));
			}
			
		}
		fclose(file_ptr);
	}
}

void print_out1to9_weights(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1, int dim2)
{	
	int length = dim1*dim2;
	int count = 0;

	// hidden_to_out_weights[45][10]

	printf("\nPrinting %s : ADAPT %d : Print %d : PrinttoFile %d : Path %s : Mode %s\n\n%s\n", name, adapt, print_to_stdout, print_to_file, path, mode, name);

	if(print_to_stdout == 1) 
	{
		count = 0;
		for(int i = 0; i < dim1; i++) 
		{
			printf("\n%d/%d\n", i+1, dim1);
			for(int j = 0; j < dim2; j++) 
			{				
				count = count + 1;
				printf("out1to9 Weights : Index %d/%d : %f\n", count, length, hidden_to_out_weights1to9[i][j]);					
			}
		}
	}		
	printf("\n");
	fflush(stdout);
	
	if(print_to_file == 1)
	{
		count = 0;
		FILE * file_ptr = NULL;
		char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for(int i = 0; i < dim1; i++) 
		{
			for(int j = 0; j < dim2; j++) 
			{
				count = count + 1;
				if(count >= length) {
					fprintf(file_ptr, "%s", gcvt(hidden_to_out_weights1to9[i][j], precision, float_buffer));	
				} 
				else 
				{
					fprintf(file_ptr, "%s\n", gcvt(hidden_to_out_weights1to9[i][j], precision, float_buffer));	
				}	
				
			}
		}

		fclose(file_ptr);

	}
	
}

void print_out1to9_biases(int adapt, const char * name, int print_to_stdout, int print_to_file, const char * path, const char * mode, int precision, int dim1)
{	
	// float convpa_bias[6];
	int count = 0;
	int length = dim1;

	printf("\nPrinting %s : ADAPT %d : Print %d : PrinttoFile %d : Path %s : Mode %s\n\n%s\n",  name, adapt, print_to_stdout, print_to_file, path, mode, name);

	if(print_to_stdout == 1) 
	{
		for (int i = 0; i < dim1; i++)
		{
			printf("%d / %d : %f\n", i+1, dim1, out_bias1to9[i]);	
		}		
	printf("\n");
	fflush(stdout);
	}

	if(print_to_file == 1)
	{
		FILE * file_ptr = NULL;
		char float_buffer[256];

		file_ptr = fopen(path,mode);
		if(file_ptr == NULL) { printf("Failed to open file %s, exiting...", name); exit(1);}

		for (int i = 0; i < dim1; i++)
		{
			count = count + 1;
			if(count >= length) {
				fprintf(file_ptr, "%s", gcvt(out_bias1to9[i], precision, float_buffer));
			}
			else
			{
				fprintf(file_ptr, "%s\n", gcvt(out_bias1to9[i], precision, float_buffer));
			}
			
		}
		fclose(file_ptr);
	}
}



/// LOAD PARAMETERS FUNCTIONS ///
void load_param_conv_weights(const char * path)
{
	printf("Loading from folder %s", path);

	const char * param_path = NULL;
	int length = 0;
	int count = 0;
	int dim1 = 0;	
	int dim2 = 0;
	int dim3 = 0;
	int dim4 = 0;
	float value = 0;
	
	float * file_mem = NULL;

	// float conv_k_weights[6][5][5], 
	param_path = path;
	dim1 = 6;
	dim2 = 5;
	dim3 = 5;
	length = dim1*dim2*dim3;

	printf("Loading %s\n", param_path);
	
	count = 0;
	
	printf("%s : prod(dims) = Length : %d x %d x %d = %d\n", param_path, dim1, dim2, dim3, length);

	file_mem = read_params(param_path, length);

	for(int i = 0; i < dim1; i++) 
	{
		for(int j = 0; j < dim2; j++) 
		{
			printf("\n%d/%d %d/%d\n", i+1, dim1, j+1, dim2);
			for(int k = 0; k < dim3; k++) 
			{

				value = *(file_mem+count);
				// printf("%s : Index %d/%d : %f\n", param_path, count, length, value);

				conv_k_weights[i][j][k] = *(file_mem+count);

				count = count + 1;
				printf("CONV1 Weights : Index %d/%d : %f\n", count, length, conv_k_weights[i][j][k]);
						
			}
		}
	}
	printf("\nCONV1 weights loaded\n\n");
}

void load_param_conv_biases(const char * path)
{
	// Setup

	const char * param_path = NULL;
	int length = 0;
	int count = 0;
	int dim1 = 0;	
	int dim2 = 0;
	int dim3 = 0;
	int dim4 = 0;
	float value = 0;
	float * file_mem = NULL;

	// float conv_k_bias[6]
	length = 6;
	param_path = path;
	printf("Loading %s\n", param_path);
	printf("%s : Loading %d entries\n", param_path, length);

	file_mem = read_params(param_path, length);
	for(int i = 0; i < length; i++) 
	{
		conv_k_bias[i] = *(file_mem+i);
		printf("%s : Index %d/%d : %f\n", param_path, i+1, length, conv_k_bias[i]);
	}
	free(file_mem);

	printf("\nCONV biases loaded\n\n");

}

void load_param_SA_weights(const char * path)
{
	// Setup

	const char * param_path = NULL;
	int length = 0;
	int count = 0;
	int dim1 = 0;	
	int dim2 = 0;
	int dim3 = 0;
	int dim4 = 0;
	float value = 0;
	float * file_mem = NULL;

	// float conva_weight[6][6];
	param_path = path;
	dim1 = 6;
	dim2 = 6;
	length = dim1*dim2;

	printf("Loading %s\n", param_path);
	
	count = 0;
	
	printf("%s : prod(dims) = Length : %d x %d = %d\n", param_path, dim1, dim2, length);

	file_mem = read_params(param_path, length);

	for(int i = 0; i < dim1; i++) 
	{
		printf("\n%d/%d \n", i+1, dim1);

		for(int j = 0; j < dim2; j++) 
		{

			value = *(file_mem+count);
			// printf("%s : Index %d/%d : %f\n", param_path, count, length, value);

			conva_weight[i][j] = *(file_mem+count);

			count = count + 1;
			printf("SA Weights : Index %d/%d : %f\n", count, length, conva_weight[i][j]);						
		}
	}
	printf("\nSA weights loaded\n\n");
	
}

void load_param_SA_biases(const char * path)
{
	// Setup

	const char * param_path = NULL;
	int length = 0;
	int count = 0;
	int dim1 = 0;	
	int dim2 = 0;
	int dim3 = 0;
	int dim4 = 0;
	float value = 0;
	float * file_mem = NULL;

	// float conva_bias[6];
	length = 6;
	param_path = path;
	printf("Loading %s\n", param_path);
	printf("%s : Loading %d entries\n", param_path, length);

	file_mem = read_params(param_path, length);
	for(int i = 0; i < length; i++) 
	{
		conva_bias[i] = *(file_mem+i);
		printf("%s : Index %d/%d : %f\n", param_path, i+1, length, conva_bias[i]);
	}
	free(file_mem);

	printf("\nSA biases loaded\n\n");
	
}

void load_param_PA_weights(const char * path)
{
	// Setup

	const char * param_path = NULL;
	int length = 0;
	int count = 0;
	int dim1 = 0;	
	int dim2 = 0;
	int dim3 = 0;
	int dim4 = 0;
	float value = 0;
	float * file_mem = NULL;

	// float convpa_weight[6];
	length = 6;
	param_path = path;
	printf("Loading %s\n", param_path);
	printf("%s : Loading %d entries\n", param_path, length);

	file_mem = read_params(param_path, length);
	for(int i = 0; i < length; i++) 
	{
		convpa_weight[i] = *(file_mem+i);
		printf("%s : Index %d/%d : %f\n", param_path, i+1, length, convpa_weight[i]);
	}
	free(file_mem);

	printf("\nPA weights loaded\n\n");

}

void load_param_PA_biases(const char * path)
{
	// Setup

	const char * param_path = NULL;
	int length = 0;
	int count = 0;
	int dim1 = 0;	
	int dim2 = 0;
	int dim3 = 0;
	int dim4 = 0;
	float value = 0;
	float * file_mem = NULL;

	// float convpa_bias[6];
	length = 6;
	param_path = path;
	printf("Loading %s\n", param_path);
	printf("%s : Loading %d entries\n", param_path, length);

	file_mem = read_params(param_path, length);
	for(int i = 0; i < length; i++) 
	{
		convpa_bias[i] = *(file_mem+i);
		printf("%s : Index %d/%d : %f\n", param_path, i+1, length, convpa_bias[i]);
	}
	free(file_mem);

	printf("\nPA biases loaded\n\n");
	
}

void load_param_fc_weights(const char * path)
{
	// Setup

	const char * param_path = NULL;
	int length = 0;
	int count = 0;
	int dim1 = 0;	
	int dim2 = 0;
	int dim3 = 0;
	int dim4 = 0;
	float value = 0;
	float * file_mem = NULL;

	// max_to_hidden_weights[6][12][12][45] FC WEIGHT
	param_path = path;
	dim1 = 6;
	dim2 = 12;
	dim3 = 12;
	dim4 = 45;

	printf("Loading %s\n", param_path);
	
	count = 0;
	length = dim1*dim2*dim3*dim4;
	printf("%s : prod(dims) = Length : %d x %d x %d x %d = %d\n", param_path, dim1, dim2, dim3, dim4, length);

	file_mem = read_params(param_path, length);

	for(int i = 0; i < dim1; i++) 
	{
		for(int j = 0; j < dim2; j++) 
		{
			for(int k = 0; k < dim3; k++) 
			{
				printf("\n%d/%d %d/%d %d/%d\n", i+1, dim1, j+1, dim2, k+1, dim3);
				for(int l = 0; l < dim4; l++) 
				{

					value = *(file_mem+count);
					// printf("%s : Index %d/%d : %f\n", param_path, count, length, value);
					max_to_hidden_weights[i][j][k][l] = *(file_mem+count);

					count = count + 1;
					printf("max_to_hidden_weights FC Weights : Index %d/%d : %f\n", count, length, max_to_hidden_weights[i][j][k][l]);
					if(count == length) {
						printf("\nmax_to_hidden_weights FC weights loaded\n");
					}

				}				
			}
		}
	}
}

void load_param_fcpa_weights(const char * path)
{
	// Setup

	const char * param_path = NULL;
	int length = 0;
	int count = 0;
	int dim1 = 0;	
	int dim2 = 0;
	int dim3 = 0;
	int dim4 = 0;
	float value = 0;
	float * file_mem = NULL;

/*// max_to_hidden_weights_pa[6][14][14][45] FC WEIGHT
	param_path = path;
	dim1 = 6;
	dim2 = 14;
	dim3 = 14;
	dim4 = 45;

	printf("Loading %s\n", param_path);
	
	count = 0;
	length = dim1*dim2*dim3*dim4;
	printf("%s : prod(dims) = Length : %d x %d x %d x %d = %d\n", param_path, dim1, dim2, dim3, dim4, length);

	file_mem = read_params(param_path, length);

	for(int i = 0; i < dim1; i++) 
	{
		for(int j = 0; j < dim2; j++) 
		{
			for(int k = 0; k < dim3; k++) 
			{
				printf("\n%d/%d %d/%d %d/%d\n", i+1, dim1, j+1, dim2, k+1, dim3);
				for(int l = 0; l < dim4; l++) 
				{

					value = *(file_mem+count);
					printf("%s : Index %d/%d : %f\n", param_path, count, length, value);
					max_to_hidden_weights[i][j][k][l] = *(file_mem+count);

					count = count + 1;
					printf("FC Weights : Index %d/%d : %f\n", count, length, max_to_hidden_weights[i][j][k][l]);
					if(count == length) {
						printf("\nFC weights loaded\n");
					}

				}				
			}
		}
	}*/
}

void load_param_fc_biases(const char * path)
{
	// Setup

	const char * param_path = NULL;
	int length = 0;
	int count = 0;
	int dim1 = 0;	
	int dim2 = 0;
	int dim3 = 0;
	int dim4 = 0;
	float value = 0;
	float * file_mem = NULL;

	// hidden_bias[45]
	length = 45;
	param_path = path;
	printf("Loading %s\n", param_path);
	printf("%s : Loading %d entries\n", param_path, length);

	file_mem = read_params(param_path, length);
	for(int i = 0; i < length; i++) 
	{
		hidden_bias[i] = *(file_mem+i);
		printf("%s : Index %d/%d : %f\n", param_path, i+1, length, hidden_bias[i]);
	}
	free(file_mem);

	printf("\nFC biases loaded\n\n");
	
}



void load_param_out_weights(const char * path)
{
	// Setup

	const char * param_path = NULL;
	int length = 0;
	int count = 0;
	int dim1 = 0;	
	int dim2 = 0;
	int dim3 = 0;
	int dim4 = 0;
	float value = 0;
	float * file_mem = NULL;

	// hidden_to_out_weights[45][10]
	param_path = path;
	dim1 = 45;
	dim2 = 10;
	length = dim1*dim2;

	printf("Loading %s\n", param_path);
	
	count = 0;
	
	printf("%s : prod(dims) = Length : %d x %d = %d\n", param_path, dim1, dim2, length);

	file_mem = read_params(param_path, length);

	for(int i = 0; i < dim1; i++) 
	{
		printf("\n%d/%d \n", i+1, dim1);

		for(int j = 0; j < dim2; j++) 
		{

			value = *(file_mem+count);
			// printf("%s : Index %d/%d : %f\n", param_path, count, length, value);

			hidden_to_out_weights[i][j] = *(file_mem+count);

			count = count + 1;
			printf("FC to OUT : Index %d/%d : %f\n", count, length, hidden_to_out_weights[i][j]);						
		}
	}
	printf("\nFC to OUT weights loaded\n\n");
	
}

void load_param_out_biases(const char * path)
{
	// Setup

	const char * param_path = NULL;
	int length = 0;
	int count = 0;
	int dim1 = 0;	
	int dim2 = 0;
	int dim3 = 0;
	int dim4 = 0;
	float value = 0;
	float * file_mem = NULL;

	// out_bias[10]
	length = 10;
	param_path = path;
	printf("Loading %s\n", param_path);
	printf("%s : Loading %d entries\n", param_path, length);

	file_mem = read_params(param_path, length);
	for(int i = 0; i < length; i++) 
	{
		out_bias[i] = *(file_mem+i);
		printf("%s : Index %d/%d : %f\n", param_path, i+1, length, out_bias[i]);
	}
	free(file_mem);

	printf("\nOUT biases loaded\n\n");
	
}

void load_param_out1to9_weights(const char * path) 
{
	// Setup

	const char * param_path = NULL;
	int length = 0;
	int count = 0;
	int dim1 = 0;	
	int dim2 = 0;
	int dim3 = 0;
	int dim4 = 0;
	float value = 0;
	float * file_mem = NULL;

	// hidden_to_out1to9_weights[45][9]

}

void load_param_out1to9_biases(const char * path) 
{
	// Setup

	const char * param_path = NULL;
	int length = 0;
	int count = 0;
	int dim1 = 0;	
	int dim2 = 0;
	int dim3 = 0;
	int dim4 = 0;
	float value = 0;
	float * file_mem = NULL;

	// // out_bias1to9[9]
	// length = 9;
	// param_path = path;
	// printf("Loading %s\n", param_path);
	// printf("%s : Loading %d entries\n", param_path, length);

	// file_mem = read_params(param_path, length);
	// for(int i = 0; i < length; i++) 
	// {
	// 	out_bias1to9[i] = *(file_mem+i);
	// 	printf("%s : Index %d/%d : %f\n", param_path, i+1, length, out_bias1to9[i]);
	// }
	// free(file_mem);

}
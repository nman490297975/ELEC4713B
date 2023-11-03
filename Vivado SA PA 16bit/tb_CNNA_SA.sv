`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 14.10.2023 15:10:01
// Design Name: 
// Module Name: tb_CNNA_SA
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module tb_CNNA_SA;

/*------------------------------------------------------------------------------
--  Module Parameters
------------------------------------------------------------------------------*/
//// Hardware Parameters ////
parameter int BIT_WIDTH = 11;
parameter int BW = BIT_WIDTH + 2 - 1;

// Inputs
parameter int SA_FMAP_CHANNELS = 6;  
parameter int SA_FMAP_IMAGE_SIZE = 24*24; 
parameter int SA_FMAP_LENGTH = SA_FMAP_CHANNELS * SA_FMAP_IMAGE_SIZE; 

// Kernels
parameter int SA_KERNELS = 1; //6; //Want to do 1 channel at a time
parameter int SA_KERNELS_TOTAL = 6;

// Neuron Count
parameter int SA_LENGTH = SA_KERNELS * SA_FMAP_IMAGE_SIZE; // 6x576 = 3456

// Output
parameter int SA_OUTPUT_Z_LENGTH = SA_LENGTH; // 1x576 = 3456

// Parameters
parameter int SA_WEIGHTS_LENGTH = SA_FMAP_CHANNELS * SA_KERNELS;
parameter int SA_BIASES_LENGTH = SA_KERNELS;
parameter int SA_POOL_ERROR_LENGTH = SA_OUTPUT_Z_LENGTH;


//// Test Parameters ////
parameter int TEST_LENGTH = 4;

// FP Inputs //
parameter int TEST_SA_FMAP_LENGTH = TEST_LENGTH * SA_FMAP_LENGTH;
parameter int TEST_SA_WEIGHTS_LENGTH = TEST_LENGTH * SA_WEIGHTS_LENGTH * SA_KERNELS_TOTAL;
parameter int TEST_SA_BIASES_LENGTH = TEST_LENGTH * SA_BIASES_LENGTH * SA_KERNELS_TOTAL;

// BP Inputs //
parameter int TEST_SA_DERIV_ACTV_LENGTH = TEST_LENGTH * SA_OUTPUT_Z_LENGTH * SA_KERNELS_TOTAL; 
parameter int TEST_SA_POOL_ERROR_LENGTH = TEST_LENGTH * SA_POOL_ERROR_LENGTH * SA_KERNELS_TOTAL; //temp0 error 576*6
parameter int TEST_SA_DELTA_ERROR_LENGTH = TEST_LENGTH * SA_LENGTH * SA_KERNELS_TOTAL;
parameter int TEST_SA_DELTA_WEIGHTS_LENGTH = TEST_SA_WEIGHTS_LENGTH;
parameter int TEST_SA_DELTA_BIASES_LENGTH = TEST_SA_BIASES_LENGTH;

/*------------------------------------------------------------------------------
--  Input / Output Control signals for Adapter
------------------------------------------------------------------------------*/

reg clk = 0;
reg rst;
logic do_fp, do_bp;

logic done_FP, done_BP;

/*------------------------------------------------------------------------------
--  I/O Data Signals for Adapter
------------------------------------------------------------------------------*/

logic [(SA_LENGTH -1):0]                        [BW:0]   fp_SA_fmap;        // 6*24*24  = 3456
//logic [(SA_LENGTH -1):0]                        [BW:0]   fp_SA_fmap_sum;
logic [(SA_WEIGHTS_LENGTH -1):0]                [BW:0]   fp_SA_weights;     // Kernels * FMAP channels = 6*6 = 36
logic [(SA_BIASES_LENGTH -1):0]                 [BW:0]   fp_SA_biases;      // 1 per kernel = 6 

logic [(SA_OUTPUT_Z_LENGTH -1):0]               [BW:0]   bp_SA_pool_error;  // 6*24*24 (error from actv(conv+adapt) layer)

logic [(SA_OUTPUT_Z_LENGTH -1):0]               [BW:0]   fp_SA_output_z;    // 6*24*24  = 3456

logic [(SA_OUTPUT_Z_LENGTH -1):0]               [BW:0]   bp_SA_deriv_actv;  // 6*24*24  = 3456
logic [(SA_LENGTH -1):0]                        [BW:0]   bp_SA_delta_error; // 6*24*24  = 3456 (error propogated backwards from adapter layer)
logic [(SA_FMAP_CHANNELS * SA_KERNELS -1):0]    [BW:0]   bp_SA_delta_weights; // Kernels * FMAP channels = 6*6 = 36
logic [(SA_KERNELS -1):0]                       [BW:0]   bp_SA_delta_biases;  // 1 per kernel = 6  

/*------------------------------------------------------------------------------
--  Memory Signals (ONLY INPUTS WE NEED i.e NOT COMPARING)
------------------------------------------------------------------------------*/

logic [BW:0] mem_fp_SA_fmap          [(TEST_SA_FMAP_LENGTH -1):0]; 
//logic [BW:0] mem_fp_SA_fmap_sum      [(TEST_SA_FMAP_LENGTH -1):0]; 

logic [BW:0] mem_fp_SA_weights       [(TEST_SA_WEIGHTS_LENGTH -1):0]; 
logic [BW:0] mem_fp_SA_biases        [(TEST_SA_BIASES_LENGTH -1):0]; 

logic [BW:0] mem_bp_SA_pool_error    [(TEST_SA_POOL_ERROR_LENGTH -1):0]; 


/*------------------------------------------------------------------------------
--  Data signals for testbench
------------------------------------------------------------------------------*/
logic               start;
logic [3:0]         test_counter;
logic [3:0]         channel; 
logic [3:0]         k, l, n;
logic [15:0]         m; 
logic               write_fmap, write_weight;
logic               finished;


/*------------------------------------------------------------------------------
--  I/O Data Signals for RAM
------------------------------------------------------------------------------*/

integer i = 0;
integer j = 0;
shortreal real_number1;
integer fd_fp_SA_adapter_z;
integer fd_fp_SA_adapter_z_bychan;
integer fd_bp_SA_deriv_actv;
integer fd_bp_SA_delta_error;
integer fd_bp_SA_delta_weights;
integer fd_bp_SA_delta_biases;
logic new_input_en;
logic write_to_file;


/*------------------------------------------------------------------------------
--  Generate modules
------------------------------------------------------------------------------*/

cnn_topmodule top_module_SA (
    .clk(clk), // Control //
    .rst(rst),
    .do_fp(do_fp),
    .do_bp(do_bp),
    .done_FP(done_FP),
    .done_BP(done_BP),

    .fp_fmap(fp_SA_fmap), // FP //
    //.fp_fmap_current(fp_SA_fmap_sum),
    .weights_SA(fp_SA_weights),
    .biases_SA(fp_SA_biases),

    .adapter_z(fp_SA_output_z),

    .error_IN(bp_SA_pool_error), // BP //

    .acc_deriv_SA(bp_SA_deriv_actv),
    .error_OUT_SA(bp_SA_delta_error),
    .bpWchange_SA(bp_SA_delta_weights),
    .bpBchange_SA(bp_SA_delta_biases)
);

integer fd_fp_SA_biases;

/*------------------------------------------------------------------------------
--  Start Simulation
------------------------------------------------------------------------------*/

// Clock generation
initial begin
    forever #5 clk <= ~clk;
end

// Reset generation
initial begin

    /*------------------------------------------------------------------------------
    --  Read from memory into RAM (ONLY INPUTS WE NEED i.e NOT COMPARING)
    ------------------------------------------------------------------------------*/
    // FP //
    //$readmemh("sa.conv1.fmap_hex.mem", mem_fp_SA_fmap_sum);
    $readmemh("sa.conv1.fmap_hex.mem", mem_fp_SA_fmap);
    $readmemh("sa.weights_hex.mem", mem_fp_SA_weights);
    $readmemh("sa.biases_hex.mem", mem_fp_SA_biases);

    // BP //
    $readmemh("sa.maxpool_error_hex.mem", mem_bp_SA_pool_error); //How maxpool is read in affects how to calculate bp_SA_delta_error//

    /*------------------------------------------------------------------------------
    --  Prepare write to file 
    ------------------------------------------------------------------------------*/
    fd_fp_SA_biases = $fopen("test_write.txt", "w");
    $display("fd_fp_SA_biases: %d ", fd_fp_SA_biases);
    $fwrite(fd_fp_SA_biases,"%s","File cleared\n");
    $fclose(fd_fp_SA_biases);   
    fd_fp_SA_biases = $fopen("test_write.txt", "a");
    $fwrite(fd_fp_SA_biases,"%s","Test write begins\n");
    $fwrite(fd_fp_SA_biases,"%s","Test write DONE\n");
    $fclose(fd_fp_SA_biases);
    
    fd_fp_SA_adapter_z_bychan = $fopen("adapter_z_bychan.txt", "w");
    $display("fd_fp_SA_adapter_z_bychan: %d ", fd_fp_SA_adapter_z_bychan);
    $fwrite(fd_fp_SA_adapter_z_bychan,"%s","File cleared\n");
    $fclose(fd_fp_SA_adapter_z_bychan);  
    
    fd_fp_SA_adapter_z = $fopen("adapter_z.txt", "w");
    $display("fd_fp_SA_adapter_z: %d ", fd_fp_SA_adapter_z);
    $fwrite(fd_fp_SA_adapter_z,"%s","File cleared\n");
    $fclose(fd_fp_SA_adapter_z);  
    
    fd_bp_SA_deriv_actv = $fopen("deriv_actv.txt", "w");
    $display("fd_bp_SA_deriv_actv: %d ", fd_bp_SA_deriv_actv);
    $fwrite(fd_bp_SA_deriv_actv,"%s","File cleared\n");
    $fclose(fd_bp_SA_deriv_actv);  

    fd_bp_SA_delta_error = $fopen("delta_error.txt", "w");
    $display("fd_bp_SA_delta_error: %d ", fd_bp_SA_delta_error);
    $fwrite(fd_bp_SA_delta_error,"%s","File cleared\n");
    $fclose(fd_bp_SA_delta_error);

    fd_bp_SA_delta_weights = $fopen("delta_weights.txt", "w");
    $display("fd_bp_SA_delta_weights: %d ", fd_bp_SA_delta_weights);
    $fwrite(fd_bp_SA_delta_weights,"%s","File cleared\n");
    $fclose(fd_bp_SA_delta_weights);

    fd_bp_SA_delta_biases = $fopen("delta_biases.txt", "w");
    $display("fd_bp_SA_delta_biases: %d ", fd_bp_SA_delta_biases);
    $fwrite(fd_bp_SA_delta_biases,"%s","File cleared\n");
    $fclose(fd_bp_SA_delta_biases);


    /*------------------------------------------------------------------------------
    --  Start Processing
    ------------------------------------------------------------------------------*/
    rst <= 1;
    start <= 0;

    repeat (2) @ (posedge clk);

    repeat (1) @ (posedge clk) begin
        rst <= 0;
    end

    repeat (2) @ (posedge clk);

    repeat (1) @ (posedge clk) begin
        start <= 1;
    end 

end

/*------------------------------------------------------------------------------
    --  Control Signal
------------------------------------------------------------------------------*/
enum logic [3:0] {idle, read_data, fp_sum, fp, bp_error, bp_bias, bp_weight, write, new_data, finish} cs, ns;

always_ff @(posedge clk, posedge rst)
    if (rst) cs <= idle;
    else cs <= ns;

always_ff @(posedge clk, posedge rst) begin
    if (rst) begin
        finished <= 0;
        test_counter <= 0;
        channel <= 0;
        k <= 0;
        l <= 0;
        m <= 0;
        n <= 0;
    end
    else begin
        case(cs)

            fp_sum: begin
                k <= (k < SA_KERNELS_TOTAL - 1) ? (k + 1) : channel;
                l <= l + 1;
                write_fmap <= 1;
            end

            fp: begin
                write_fmap <= 0;
            end

            bp_error: begin
                k <= 0;
                l <= 0;
            end

            bp_bias: begin
                m <= (m < SA_LENGTH - 1) ? (m + 1) : 0;
                write_fmap <= (m == SA_LENGTH - 1) ? 1 : 0;
            end

            bp_weight: begin
                m <= (m < SA_LENGTH - 1) ? (m + 1) : 0;
                n <= (m == SA_LENGTH - 1) ? ((n < SA_KERNELS_TOTAL) ? (n + 1) : 0) : n;
                k <= (m == SA_LENGTH - 2) ? ((k < SA_KERNELS_TOTAL - 1) ? (k + 1) : 0) : k;
                write_fmap <= (m == SA_LENGTH - 2) ? 1 : 0;
            end

            write: begin
                //test_counter <= test_counter + 1;
                channel <= channel + 1;
                n <= 0;
                k <= 0;
                test_counter <= (channel < SA_KERNELS_TOTAL - 1) ? test_counter : test_counter + 1;
            end

            new_data: begin
                //test_counter <= test_counter + 1;
                channel <= 0;
            end
       endcase
    end
end

// Next state and output logic
always_comb begin
    do_fp = 0;
    do_bp = 0;
    new_input_en = 0;
    write_to_file = 0;
    write_weight = 0;
    case (cs) 
        idle: begin
            ns = start ? read_data : idle;
        end
        read_data: begin
            new_input_en = 1;
            ns = fp_sum;
        end
        fp_sum: begin
            do_fp = 1;
            ns = (l < SA_KERNELS_TOTAL - 1) ? fp_sum : fp;
        end
        fp: begin
            ns = done_FP ? bp_error : fp;
        end
        bp_error: begin
            do_bp = 1;
            ns = bp_bias;
        end
        bp_bias: begin
            do_bp = 1;
            ns = (m  == SA_LENGTH - 1) ? bp_weight : bp_bias;
        end
        bp_weight: begin
            ns = (n == SA_KERNELS_TOTAL) ? write: bp_weight; 
        end
        write: begin
            //test_counter = test_counter + 1;
            write_to_file = 1;
            ns = (channel < SA_KERNELS_TOTAL - 1) ? read_data : new_data;
        end
        new_data: begin
            ns = (test_counter < TEST_LENGTH) ? idle : finish;
        end
        finish: begin
            finished = 1;
            $finish;
        end
    endcase
end 

/*------------------------------------------------------------------------------
    --  Write to file
------------------------------------------------------------------------------*/
always @(posedge clk) begin //write_to_file

    if (write_to_file == 1) begin

        // Print adapter z by channel logic
        fd_fp_SA_adapter_z_bychan = $fopen("adapter_z_bychan.txt", "a");
        for (i=0; i<SA_FMAP_IMAGE_SIZE; i = i+1) begin // 576

            real_number1 = $bitstoshortreal(fp_SA_output_z[i]); //[0...3455]
            $fwrite(fd_fp_SA_adapter_z_bychan,"%f\n",real_number1);

        end
        //$fwrite(fd_fp_SA_adapter_z_bychan,"%s","Test write DONE\n");
        $fclose(fd_fp_SA_adapter_z_bychan);  

        // Print adapter z logic
        fd_fp_SA_adapter_z = $fopen("adapter_z.txt", "a");
        for (j=0; j<SA_FMAP_IMAGE_SIZE; j = j+1) begin //576

            real_number1 = $bitstoshortreal(fp_SA_output_z[j]); //[0...3455]
            $fwrite(fd_fp_SA_adapter_z,"%f\n",real_number1);

        end        
        //$fwrite(fd_fp_SA_adapter_z,"%s","Test write DONE\n");
        $fclose(fd_fp_SA_adapter_z); 

        // Print deriv_actv
        fd_bp_SA_deriv_actv = $fopen("deriv_actv.txt", "a");
        for (j=0; j<SA_FMAP_IMAGE_SIZE; j = j+1) begin //576

            real_number1 = $bitstoshortreal(bp_SA_deriv_actv[j]); //[0...3455]
            $fwrite(fd_bp_SA_deriv_actv,"%f\n",real_number1);

        end        
        //$fwrite(fd_bp_SA_deriv_actv,"%s","Test write DONE\n");
        $fclose(fd_bp_SA_deriv_actv); 

        // Print delta_error
        fd_bp_SA_delta_error = $fopen("delta_error.txt", "a");
        for (j=0; j<SA_FMAP_IMAGE_SIZE; j = j+1) begin //576

            real_number1 = $bitstoshortreal(bp_SA_delta_error[j]); //[0...3455]
            $fwrite(fd_bp_SA_delta_error,"%f\n",real_number1);

        end        
        //$fwrite(fd_bp_SA_delta_error,"%s","Test write DONE\n");
        $fclose(fd_bp_SA_delta_error); 

        // Print delta_weights
        fd_bp_SA_delta_weights = $fopen("delta_weights.txt", "a");
        for (j=0; j<SA_FMAP_CHANNELS; j = j+1) begin //576

            real_number1 = $bitstoshortreal(bp_SA_delta_weights[j]); //[0...36]
            $fwrite(fd_bp_SA_delta_weights,"%f\n",real_number1);

        end        
        //$fwrite(fd_bp_SA_delta_weights,"%s","Test write DONE\n");
        $fclose(fd_bp_SA_delta_weights); 

        // Print delta_biases
        fd_bp_SA_delta_biases = $fopen("delta_biases.txt", "a");

        real_number1 = $bitstoshortreal(bp_SA_delta_biases); //[0...6]
        $fwrite(fd_bp_SA_delta_biases,"%f\n",real_number1);
       
        //$fwrite(fd_bp_SA_delta_biases,"%s","Test write DONE\n");
        $fclose(fd_bp_SA_delta_biases); 
        
        $display("test_counter_updated: %d test_channel_updated %d", test_counter, channel);

    end
end

/*------------------------------------------------------------------------------
    --  Read in new input Signal
------------------------------------------------------------------------------*/
always @(new_input_en) begin

    if (new_input_en == 1) begin

        // // Load FMAPs
        // for (i=0; i<SA_FMAP_LENGTH; i = i+1) begin
        // fp_SA_fmap[i] = mem_fp_SA_fmap[test_counter*SA_FMAP_LENGTH + i];
        // end

        //Load FMAPs
        for (i=0; i<SA_POOL_ERROR_LENGTH; i = i+1) begin
          fp_SA_fmap[i] = mem_fp_SA_fmap[test_counter*SA_POOL_ERROR_LENGTH*SA_KERNELS_TOTAL + k*576 + i];
        end

        // Load Weights
        for (i=0; i<SA_WEIGHTS_LENGTH; i = i+1) begin
        fp_SA_weights[i] = mem_fp_SA_weights[test_counter*SA_WEIGHTS_LENGTH*SA_KERNELS_TOTAL + channel*6 + i];
        end

        // Load Biases
        for (i=0; i<SA_BIASES_LENGTH; i = i+1) begin
        fp_SA_biases[i] = mem_fp_SA_biases[test_counter*SA_BIASES_LENGTH*SA_KERNELS_TOTAL+ channel + i];
        end

        //Load pool error
        for (i=0; i<SA_POOL_ERROR_LENGTH; i = i+1) begin
          bp_SA_pool_error[i] = mem_bp_SA_pool_error[test_counter*SA_POOL_ERROR_LENGTH*SA_KERNELS_TOTAL+ channel*576 + i];
        end

    end
end

always @(posedge clk) begin

    //Load FMAPs
    if (write_fmap == 1) begin
        for (i=0; i<SA_POOL_ERROR_LENGTH; i = i+1) begin
            fp_SA_fmap[i] = mem_fp_SA_fmap[test_counter*SA_POOL_ERROR_LENGTH*SA_KERNELS_TOTAL + k*576 + i];
        end
    end

end

endmodule

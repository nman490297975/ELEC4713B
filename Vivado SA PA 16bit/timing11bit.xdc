#create_clock -period 10.000 -name clk -waveform {0.000 5.000} [get_ports clk]
create_clock -period 27.027 -name clk -waveform {0.000 13.514} [get_ports {clk}]

set_property HD.CLK_SRC BUFGCE_X0Y0 [get_ports clk] 
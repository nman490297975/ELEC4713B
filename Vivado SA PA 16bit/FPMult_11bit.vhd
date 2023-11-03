----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 18.10.2023 22:59:04
-- Design Name: 
-- Module Name: FPMult_11bit - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------

--------------------------------------------------------------------------------
--                          DSPBlock_11x11_comb_uid9
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_11x11_comb_uid9 is
    port (X : in  std_logic_vector(10 downto 0);
          Y : in  std_logic_vector(10 downto 0);
          R : out  std_logic_vector(21 downto 0)   );
end entity;

architecture arch of DSPBlock_11x11_comb_uid9 is
signal Mint :  std_logic_vector(21 downto 0);
signal M :  std_logic_vector(21 downto 0);
signal Rtmp :  std_logic_vector(21 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(21 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                          IntMultiplier_comb_uid5
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Martin Kumm, Florent de Dinechin, Kinga Illyes, Bogdan Popa, Bogdan Pasca, 2012
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity IntMultiplier_comb_uid5 is
    port (X : in  std_logic_vector(10 downto 0);
          Y : in  std_logic_vector(10 downto 0);
          R : out  std_logic_vector(21 downto 0)   );
end entity;

architecture arch of IntMultiplier_comb_uid5 is
   component DSPBlock_11x11_comb_uid9 is
      port ( X : in  std_logic_vector(10 downto 0);
             Y : in  std_logic_vector(10 downto 0);
             R : out  std_logic_vector(21 downto 0)   );
   end component;

signal XX_m6 :  std_logic_vector(10 downto 0);
signal YY_m6 :  std_logic_vector(10 downto 0);
signal tile_0_X :  std_logic_vector(10 downto 0);
signal tile_0_Y :  std_logic_vector(10 downto 0);
signal tile_0_output :  std_logic_vector(21 downto 0);
signal tile_0_filtered_output :  unsigned(21-0 downto 0);
signal bh7_w0_0 :  std_logic;
signal bh7_w1_0 :  std_logic;
signal bh7_w2_0 :  std_logic;
signal bh7_w3_0 :  std_logic;
signal bh7_w4_0 :  std_logic;
signal bh7_w5_0 :  std_logic;
signal bh7_w6_0 :  std_logic;
signal bh7_w7_0 :  std_logic;
signal bh7_w8_0 :  std_logic;
signal bh7_w9_0 :  std_logic;
signal bh7_w10_0 :  std_logic;
signal bh7_w11_0 :  std_logic;
signal bh7_w12_0 :  std_logic;
signal bh7_w13_0 :  std_logic;
signal bh7_w14_0 :  std_logic;
signal bh7_w15_0 :  std_logic;
signal bh7_w16_0 :  std_logic;
signal bh7_w17_0 :  std_logic;
signal bh7_w18_0 :  std_logic;
signal bh7_w19_0 :  std_logic;
signal bh7_w20_0 :  std_logic;
signal bh7_w21_0 :  std_logic;
signal tmp_bitheapResult_bh7_21 :  std_logic_vector(21 downto 0);
signal bitheapResult_bh7 :  std_logic_vector(21 downto 0);
begin
   XX_m6 <= X ;
   YY_m6 <= Y ;
   tile_0_X <= X(10 downto 0);
   tile_0_Y <= Y(10 downto 0);
   tile_0_mult: DSPBlock_11x11_comb_uid9
      port map ( X => tile_0_X,
                 Y => tile_0_Y,
                 R => tile_0_output);

   tile_0_filtered_output <= unsigned(tile_0_output(21 downto 0));
   bh7_w0_0 <= tile_0_filtered_output(0);
   bh7_w1_0 <= tile_0_filtered_output(1);
   bh7_w2_0 <= tile_0_filtered_output(2);
   bh7_w3_0 <= tile_0_filtered_output(3);
   bh7_w4_0 <= tile_0_filtered_output(4);
   bh7_w5_0 <= tile_0_filtered_output(5);
   bh7_w6_0 <= tile_0_filtered_output(6);
   bh7_w7_0 <= tile_0_filtered_output(7);
   bh7_w8_0 <= tile_0_filtered_output(8);
   bh7_w9_0 <= tile_0_filtered_output(9);
   bh7_w10_0 <= tile_0_filtered_output(10);
   bh7_w11_0 <= tile_0_filtered_output(11);
   bh7_w12_0 <= tile_0_filtered_output(12);
   bh7_w13_0 <= tile_0_filtered_output(13);
   bh7_w14_0 <= tile_0_filtered_output(14);
   bh7_w15_0 <= tile_0_filtered_output(15);
   bh7_w16_0 <= tile_0_filtered_output(16);
   bh7_w17_0 <= tile_0_filtered_output(17);
   bh7_w18_0 <= tile_0_filtered_output(18);
   bh7_w19_0 <= tile_0_filtered_output(19);
   bh7_w20_0 <= tile_0_filtered_output(20);
   bh7_w21_0 <= tile_0_filtered_output(21);

   -- Adding the constant bits 
      -- All the constant bits are zero, nothing to add

   tmp_bitheapResult_bh7_21 <= bh7_w21_0 & bh7_w20_0 & bh7_w19_0 & bh7_w18_0 & bh7_w17_0 & bh7_w16_0 & bh7_w15_0 & bh7_w14_0 & bh7_w13_0 & bh7_w12_0 & bh7_w11_0 & bh7_w10_0 & bh7_w9_0 & bh7_w8_0 & bh7_w7_0 & bh7_w6_0 & bh7_w5_0 & bh7_w4_0 & bh7_w3_0 & bh7_w2_0 & bh7_w1_0 & bh7_w0_0;
   bitheapResult_bh7 <= tmp_bitheapResult_bh7_21;
   R <= bitheapResult_bh7(21 downto 0);
end architecture;

--------------------------------------------------------------------------------
--                           IntAdder_17_comb_uid13
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2016)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X Y Cin
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_17_comb_uid13 is
    port (X : in  std_logic_vector(16 downto 0);
          Y : in  std_logic_vector(16 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(16 downto 0)   );
end entity;

architecture arch of IntAdder_17_comb_uid13 is
signal Rtmp :  std_logic_vector(16 downto 0);
begin
   Rtmp <= X + Y + Cin;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                                FPMult_11bit
--                        (FPMult_5_10_uid2_comb_uid3)
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin 2008-2021
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FPMult_11bit is
    port (X : in  std_logic_vector(5+10+2 downto 0);
          Y : in  std_logic_vector(5+10+2 downto 0);
          R : out  std_logic_vector(5+10+2 downto 0)   );
end entity;

architecture arch of FPMult_11bit is
   component IntMultiplier_comb_uid5 is
      port ( X : in  std_logic_vector(10 downto 0);
             Y : in  std_logic_vector(10 downto 0);
             R : out  std_logic_vector(21 downto 0)   );
   end component;

   component IntAdder_17_comb_uid13 is
      port ( X : in  std_logic_vector(16 downto 0);
             Y : in  std_logic_vector(16 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(16 downto 0)   );
   end component;

signal sign :  std_logic;
signal expX :  std_logic_vector(4 downto 0);
signal expY :  std_logic_vector(4 downto 0);
signal expSumPreSub :  std_logic_vector(6 downto 0);
signal bias :  std_logic_vector(6 downto 0);
signal expSum :  std_logic_vector(6 downto 0);
signal sigX :  std_logic_vector(10 downto 0);
signal sigY :  std_logic_vector(10 downto 0);
signal sigProd :  std_logic_vector(21 downto 0);
signal excSel :  std_logic_vector(3 downto 0);
signal exc :  std_logic_vector(1 downto 0);
signal norm :  std_logic;
signal expPostNorm :  std_logic_vector(6 downto 0);
signal sigProdExt :  std_logic_vector(21 downto 0);
signal expSig :  std_logic_vector(16 downto 0);
signal sticky :  std_logic;
signal guard :  std_logic;
signal round :  std_logic;
signal expSigPostRound :  std_logic_vector(16 downto 0);
signal excPostNorm :  std_logic_vector(1 downto 0);
signal finalExc :  std_logic_vector(1 downto 0);
begin
   sign <= X(15) xor Y(15);
   expX <= X(14 downto 10);
   expY <= Y(14 downto 10);
   expSumPreSub <= ("00" & expX) + ("00" & expY);
   bias <= CONV_STD_LOGIC_VECTOR(15,7);
   expSum <= expSumPreSub - bias;
   sigX <= "1" & X(9 downto 0);
   sigY <= "1" & Y(9 downto 0);
   SignificandMultiplication: IntMultiplier_comb_uid5
      port map ( X => sigX,
                 Y => sigY,
                 R => sigProd);
   excSel <= X(17 downto 16) & Y(17 downto 16);
   with excSel  select  
   exc <= "00" when  "0000" | "0001" | "0100", 
          "01" when "0101",
          "10" when "0110" | "1001" | "1010" ,
          "11" when others;
   norm <= sigProd(21);
   -- exponent update
   expPostNorm <= expSum + ("000000" & norm);
   -- significand normalization shift
   sigProdExt <= sigProd(20 downto 0) & "0" when norm='1' else
                         sigProd(19 downto 0) & "00";
   expSig <= expPostNorm & sigProdExt(21 downto 12);
   sticky <= sigProdExt(11);
   guard <= '0' when sigProdExt(10 downto 0)="00000000000" else '1';
   round <= sticky and ( (guard and not(sigProdExt(12))) or (sigProdExt(12) ))  ;
   RoundingAdder: IntAdder_17_comb_uid13
      port map ( Cin => round,
                 X => expSig,
                 Y => "00000000000000000",
                 R => expSigPostRound);
   with expSigPostRound(16 downto 15)  select 
   excPostNorm <=  "01"  when  "00",
                               "10"             when "01", 
                               "00"             when "11"|"10",
                               "11"             when others;
   with exc  select  
   finalExc <= exc when  "11"|"10"|"00",
                       excPostNorm when others; 
   R <= finalExc & sign & expSigPostRound(14 downto 0);
end architecture;
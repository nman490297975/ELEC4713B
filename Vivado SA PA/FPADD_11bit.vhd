----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 18.10.2023 22:59:04
-- Design Name: 
-- Module Name: FPADD_11bit - Behavioral
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
--                  RightShifterSticky11_by_max_13_comb_uid4
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca (2008-2011), Florent de Dinechin (2008-2019)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X S
-- Output signals: R Sticky

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity RightShifterSticky11_by_max_13_comb_uid4 is
    port (X : in  std_logic_vector(10 downto 0);
          S : in  std_logic_vector(3 downto 0);
          R : out  std_logic_vector(12 downto 0);
          Sticky : out  std_logic   );
end entity;

architecture arch of RightShifterSticky11_by_max_13_comb_uid4 is
signal ps :  std_logic_vector(3 downto 0);
signal Xpadded :  std_logic_vector(12 downto 0);
signal level4 :  std_logic_vector(12 downto 0);
signal stk3 :  std_logic;
signal level3 :  std_logic_vector(12 downto 0);
signal stk2 :  std_logic;
signal level2 :  std_logic_vector(12 downto 0);
signal stk1 :  std_logic;
signal level1 :  std_logic_vector(12 downto 0);
signal stk0 :  std_logic;
signal level0 :  std_logic_vector(12 downto 0);
begin
   ps<= S;
   Xpadded <= X&(1 downto 0 => '0');
   level4<= Xpadded;
   stk3 <= '1' when (level4(7 downto 0)/="00000000" and ps(3)='1')   else '0';
   level3 <=  level4 when  ps(3)='0'    else (7 downto 0 => '0') & level4(12 downto 8);
   stk2 <= '1' when (level3(3 downto 0)/="0000" and ps(2)='1') or stk3 ='1'   else '0';
   level2 <=  level3 when  ps(2)='0'    else (3 downto 0 => '0') & level3(12 downto 4);
   stk1 <= '1' when (level2(1 downto 0)/="00" and ps(1)='1') or stk2 ='1'   else '0';
   level1 <=  level2 when  ps(1)='0'    else (1 downto 0 => '0') & level2(12 downto 2);
   stk0 <= '1' when (level1(0 downto 0)/="0" and ps(0)='1') or stk1 ='1'   else '0';
   level0 <=  level1 when  ps(0)='0'    else (0 downto 0 => '0') & level1(12 downto 1);
   R <= level0;
   Sticky <= stk0;
end architecture;

--------------------------------------------------------------------------------
--                           IntAdder_14_comb_uid6
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

entity IntAdder_14_comb_uid6 is
    port (X : in  std_logic_vector(13 downto 0);
          Y : in  std_logic_vector(13 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(13 downto 0)   );
end entity;

architecture arch of IntAdder_14_comb_uid6 is
signal Rtmp :  std_logic_vector(13 downto 0);
begin
   Rtmp <= X + Y + Cin;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                      Normalizer_Z_15_15_15_comb_uid8
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin, (2007-2020)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X
-- Output signals: Count R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity Normalizer_Z_15_15_15_comb_uid8 is
    port (X : in  std_logic_vector(14 downto 0);
          Count : out  std_logic_vector(3 downto 0);
          R : out  std_logic_vector(14 downto 0)   );
end entity;

architecture arch of Normalizer_Z_15_15_15_comb_uid8 is
signal level4 :  std_logic_vector(14 downto 0);
signal count3 :  std_logic;
signal level3 :  std_logic_vector(14 downto 0);
signal count2 :  std_logic;
signal level2 :  std_logic_vector(14 downto 0);
signal count1 :  std_logic;
signal level1 :  std_logic_vector(14 downto 0);
signal count0 :  std_logic;
signal level0 :  std_logic_vector(14 downto 0);
signal sCount :  std_logic_vector(3 downto 0);
begin
   level4 <= X ;
   count3<= '1' when level4(14 downto 7) = (14 downto 7=>'0') else '0';
   level3<= level4(14 downto 0) when count3='0' else level4(6 downto 0) & (7 downto 0 => '0');

   count2<= '1' when level3(14 downto 11) = (14 downto 11=>'0') else '0';
   level2<= level3(14 downto 0) when count2='0' else level3(10 downto 0) & (3 downto 0 => '0');

   count1<= '1' when level2(14 downto 13) = (14 downto 13=>'0') else '0';
   level1<= level2(14 downto 0) when count1='0' else level2(12 downto 0) & (1 downto 0 => '0');

   count0<= '1' when level1(14 downto 14) = (14 downto 14=>'0') else '0';
   level0<= level1(14 downto 0) when count0='0' else level1(13 downto 0) & (0 downto 0 => '0');

   R <= level0;
   sCount <= count3 & count2 & count1 & count0;
   Count <= sCount;
end architecture;

--------------------------------------------------------------------------------
--                           IntAdder_18_comb_uid11
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

entity IntAdder_18_comb_uid11 is
    port (X : in  std_logic_vector(17 downto 0);
          Y : in  std_logic_vector(17 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(17 downto 0)   );
end entity;

architecture arch of IntAdder_18_comb_uid11 is
signal Rtmp :  std_logic_vector(17 downto 0);
begin
   Rtmp <= X + Y + Cin;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                                FPADD_11bit
--                           (FPAdd_5_10_comb_uid2)
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin, Bogdan Pasca (2010-2017)
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

entity FPADD_11bit is
    port (X : in  std_logic_vector(5+10+2 downto 0);
          Y : in  std_logic_vector(5+10+2 downto 0);
          R : out  std_logic_vector(5+10+2 downto 0)   );
end entity;

architecture arch of FPADD_11bit is
   component RightShifterSticky11_by_max_13_comb_uid4 is
      port ( X : in  std_logic_vector(10 downto 0);
             S : in  std_logic_vector(3 downto 0);
             R : out  std_logic_vector(12 downto 0);
             Sticky : out  std_logic   );
   end component;

   component IntAdder_14_comb_uid6 is
      port ( X : in  std_logic_vector(13 downto 0);
             Y : in  std_logic_vector(13 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(13 downto 0)   );
   end component;

   component Normalizer_Z_15_15_15_comb_uid8 is
      port ( X : in  std_logic_vector(14 downto 0);
             Count : out  std_logic_vector(3 downto 0);
             R : out  std_logic_vector(14 downto 0)   );
   end component;

   component IntAdder_18_comb_uid11 is
      port ( X : in  std_logic_vector(17 downto 0);
             Y : in  std_logic_vector(17 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(17 downto 0)   );
   end component;

signal excExpFracX :  std_logic_vector(16 downto 0);
signal excExpFracY :  std_logic_vector(16 downto 0);
signal swap :  std_logic;
signal eXmeY :  std_logic_vector(4 downto 0);
signal eYmeX :  std_logic_vector(4 downto 0);
signal expDiff :  std_logic_vector(4 downto 0);
signal newX :  std_logic_vector(17 downto 0);
signal newY :  std_logic_vector(17 downto 0);
signal expX :  std_logic_vector(4 downto 0);
signal excX :  std_logic_vector(1 downto 0);
signal excY :  std_logic_vector(1 downto 0);
signal signX :  std_logic;
signal signY :  std_logic;
signal EffSub :  std_logic;
signal sXsYExnXY :  std_logic_vector(5 downto 0);
signal sdExnXY :  std_logic_vector(3 downto 0);
signal fracY :  std_logic_vector(10 downto 0);
signal excRt :  std_logic_vector(1 downto 0);
signal signR :  std_logic;
signal shiftedOut :  std_logic;
signal shiftVal :  std_logic_vector(3 downto 0);
signal shiftedFracY :  std_logic_vector(12 downto 0);
signal sticky :  std_logic;
signal fracYpad :  std_logic_vector(13 downto 0);
signal EffSubVector :  std_logic_vector(13 downto 0);
signal fracYpadXorOp :  std_logic_vector(13 downto 0);
signal fracXpad :  std_logic_vector(13 downto 0);
signal cInSigAdd :  std_logic;
signal fracAddResult :  std_logic_vector(13 downto 0);
signal fracSticky :  std_logic_vector(14 downto 0);
signal nZerosNew :  std_logic_vector(3 downto 0);
signal shiftedFrac :  std_logic_vector(14 downto 0);
signal extendedExpInc :  std_logic_vector(5 downto 0);
signal updatedExp :  std_logic_vector(6 downto 0);
signal eqdiffsign :  std_logic;
signal expFrac :  std_logic_vector(17 downto 0);
signal stk :  std_logic;
signal rnd :  std_logic;
signal lsb :  std_logic;
signal needToRound :  std_logic;
signal RoundedExpFrac :  std_logic_vector(17 downto 0);
signal upExc :  std_logic_vector(1 downto 0);
signal fracR :  std_logic_vector(9 downto 0);
signal expR :  std_logic_vector(4 downto 0);
signal exExpExc :  std_logic_vector(3 downto 0);
signal excRt2 :  std_logic_vector(1 downto 0);
signal excR :  std_logic_vector(1 downto 0);
signal signR2 :  std_logic;
signal computedR :  std_logic_vector(17 downto 0);
begin
   excExpFracX <= X(17 downto 16) & X(14 downto 0);
   excExpFracY <= Y(17 downto 16) & Y(14 downto 0);
   swap <= '1' when excExpFracX < excExpFracY else '0';
   -- exponent difference
   eXmeY <= (X(14 downto 10)) - (Y(14 downto 10));
   eYmeX <= (Y(14 downto 10)) - (X(14 downto 10));
   expDiff <= eXmeY when swap = '0' else eYmeX;
   -- input swap so that |X|>|Y|
   newX <= X when swap = '0' else Y;
   newY <= Y when swap = '0' else X;
   -- now we decompose the inputs into their sign, exponent, fraction
   expX<= newX(14 downto 10);
   excX<= newX(17 downto 16);
   excY<= newY(17 downto 16);
   signX<= newX(15);
   signY<= newY(15);
   EffSub <= signX xor signY;
   sXsYExnXY <= signX & signY & excX & excY;
   sdExnXY <= excX & excY;
   fracY <= "00000000000" when excY="00" else ('1' & newY(9 downto 0));
   -- Exception management logic
   with sXsYExnXY  select  
   excRt <= "00" when "000000"|"010000"|"100000"|"110000",
      "01" when "000101"|"010101"|"100101"|"110101"|"000100"|"010100"|"100100"|"110100"|"000001"|"010001"|"100001"|"110001",
      "10" when "111010"|"001010"|"001000"|"011000"|"101000"|"111000"|"000010"|"010010"|"100010"|"110010"|"001001"|"011001"|"101001"|"111001"|"000110"|"010110"|"100110"|"110110", 
      "11" when others;
   signR<= '0' when (sXsYExnXY="100000" or sXsYExnXY="010000") else signX;
   shiftedOut <= '1' when (expDiff > 12) else '0';
   shiftVal <= expDiff(3 downto 0) when shiftedOut='0' else CONV_STD_LOGIC_VECTOR(13,4);
   RightShifterComponent: RightShifterSticky11_by_max_13_comb_uid4
      port map ( S => shiftVal,
                 X => fracY,
                 R => shiftedFracY,
                 Sticky => sticky);
   fracYpad <= "0" & shiftedFracY;
   EffSubVector <= (13 downto 0 => EffSub);
   fracYpadXorOp <= fracYpad xor EffSubVector;
   fracXpad <= "01" & (newX(9 downto 0)) & "00";
   cInSigAdd <= EffSub and not sticky; -- if we subtract and the sticky was one, some of the negated sticky bits would have absorbed this carry 
   fracAdder: IntAdder_14_comb_uid6
      port map ( Cin => cInSigAdd,
                 X => fracXpad,
                 Y => fracYpadXorOp,
                 R => fracAddResult);
   fracSticky<= fracAddResult & sticky; 
   LZCAndShifter: Normalizer_Z_15_15_15_comb_uid8
      port map ( X => fracSticky,
                 Count => nZerosNew,
                 R => shiftedFrac);
   extendedExpInc<= ("0" & expX) + '1';
   updatedExp <= ("0" &extendedExpInc) - ("000" & nZerosNew);
   eqdiffsign <= '1' when nZerosNew="1111" else '0';
   expFrac<= updatedExp & shiftedFrac(13 downto 3);
   stk<= shiftedFrac(2) or shiftedFrac(1) or shiftedFrac(0);
   rnd<= shiftedFrac(3);
   lsb<= shiftedFrac(4);
   needToRound<= '1' when (rnd='1' and stk='1') or (rnd='1' and stk='0' and lsb='1')
  else '0';
   roundingAdder: IntAdder_18_comb_uid11
      port map ( Cin => needToRound,
                 X => expFrac,
                 Y => "000000000000000000",
                 R => RoundedExpFrac);
   -- possible update to exception bits
   upExc <= RoundedExpFrac(17 downto 16);
   fracR <= RoundedExpFrac(10 downto 1);
   expR <= RoundedExpFrac(15 downto 11);
   exExpExc <= upExc & excRt;
   with exExpExc  select  
   excRt2<= "00" when "0000"|"0100"|"1000"|"1100"|"1001"|"1101",
      "01" when "0001",
      "10" when "0010"|"0110"|"1010"|"1110"|"0101",
      "11" when others;
   excR <= "00" when (eqdiffsign='1' and EffSub='1'  and not(excRt="11")) else excRt2;
   signR2 <= '0' when (eqdiffsign='1' and EffSub='1') else signR;
   computedR <= excR & signR2 & expR & fracR;
   R <= computedR;
end architecture;
----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 07.10.2023 15:39:50
-- Design Name: 
-- Module Name: IEEEFPFMA_8_23_comb_uid2 - Behavioral
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
--                     RightShifter24_by_max_76_comb_uid4
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca (2008-2011), Florent de Dinechin (2008-2019)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X S
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity RightShifter24_by_max_76_comb_uid4 is
    port (X : in  std_logic_vector(23 downto 0);
          S : in  std_logic_vector(6 downto 0);
          R : out  std_logic_vector(99 downto 0)   );
end entity;

architecture arch of RightShifter24_by_max_76_comb_uid4 is
signal ps :  std_logic_vector(6 downto 0);
signal level0 :  std_logic_vector(23 downto 0);
signal level1 :  std_logic_vector(24 downto 0);
signal level2 :  std_logic_vector(26 downto 0);
signal level3 :  std_logic_vector(30 downto 0);
signal level4 :  std_logic_vector(38 downto 0);
signal level5 :  std_logic_vector(54 downto 0);
signal level6 :  std_logic_vector(86 downto 0);
signal level7 :  std_logic_vector(150 downto 0);
begin
   ps<= S;
   level0<= X;
   level1 <=  (0 downto 0 => '0') & level0 when ps(0) = '1' else    level0 & (0 downto 0 => '0');
   level2 <=  (1 downto 0 => '0') & level1 when ps(1) = '1' else    level1 & (1 downto 0 => '0');
   level3 <=  (3 downto 0 => '0') & level2 when ps(2) = '1' else    level2 & (3 downto 0 => '0');
   level4 <=  (7 downto 0 => '0') & level3 when ps(3) = '1' else    level3 & (7 downto 0 => '0');
   level5 <=  (15 downto 0 => '0') & level4 when ps(4) = '1' else    level4 & (15 downto 0 => '0');
   level6 <=  (31 downto 0 => '0') & level5 when ps(5) = '1' else    level5 & (31 downto 0 => '0');
   level7 <=  (63 downto 0 => '0') & level6 when ps(6) = '1' else    level6 & (63 downto 0 => '0');
   R <= level7(150 downto 51);
end architecture;

--------------------------------------------------------------------------------
--                              LZC_52_comb_uid6
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin, Bogdan Pasca (2007)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: I
-- Output signals: O

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity LZC_52_comb_uid6 is
    port (I : in  std_logic_vector(51 downto 0);
          O : out  std_logic_vector(5 downto 0)   );
end entity;

architecture arch of LZC_52_comb_uid6 is
signal level6 :  std_logic_vector(62 downto 0);
signal digit5 :  std_logic;
signal level5 :  std_logic_vector(30 downto 0);
signal digit4 :  std_logic;
signal level4 :  std_logic_vector(14 downto 0);
signal digit3 :  std_logic;
signal level3 :  std_logic_vector(6 downto 0);
signal digit2 :  std_logic;
signal level2 :  std_logic_vector(2 downto 0);
signal lowBits :  std_logic_vector(1 downto 0);
signal outHighBits :  std_logic_vector(3 downto 0);
begin
   -- pad input to the next power of two minus 1
   level6 <= I & "11111111111";
   -- Main iteration for large inputs
   digit5<= '1' when level6(62 downto 31) = "00000000000000000000000000000000" else '0';
   level5<= level6(30 downto 0) when digit5='1' else level6(62 downto 32);
   digit4<= '1' when level5(30 downto 15) = "0000000000000000" else '0';
   level4<= level5(14 downto 0) when digit4='1' else level5(30 downto 16);
   digit3<= '1' when level4(14 downto 7) = "00000000" else '0';
   level3<= level4(6 downto 0) when digit3='1' else level4(14 downto 8);
   digit2<= '1' when level3(6 downto 3) = "0000" else '0';
   level2<= level3(2 downto 0) when digit2='1' else level3(6 downto 4);
   -- Finish counting with one LUT
   with level2  select  lowBits <= 
      "11" when "000",
      "10" when "001",
      "01" when "010",
      "01" when "011",
      "00" when others;
   outHighBits <= digit5 & digit4 & digit3 & digit2 & "";
   O <= outHighBits & lowBits ;
end architecture;

--------------------------------------------------------------------------------
--                     LeftShifter76_by_max_75_comb_uid8
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca (2008-2011), Florent de Dinechin (2008-2019)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X S
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity LeftShifter76_by_max_75_comb_uid8 is
    port (X : in  std_logic_vector(75 downto 0);
          S : in  std_logic_vector(6 downto 0);
          R : out  std_logic_vector(150 downto 0)   );
end entity;

architecture arch of LeftShifter76_by_max_75_comb_uid8 is
signal ps :  std_logic_vector(6 downto 0);
signal level0 :  std_logic_vector(75 downto 0);
signal level1 :  std_logic_vector(76 downto 0);
signal level2 :  std_logic_vector(78 downto 0);
signal level3 :  std_logic_vector(82 downto 0);
signal level4 :  std_logic_vector(90 downto 0);
signal level5 :  std_logic_vector(106 downto 0);
signal level6 :  std_logic_vector(138 downto 0);
signal level7 :  std_logic_vector(202 downto 0);
begin
   ps<= S;
   level0<= X;
   level1<= level0 & (0 downto 0 => '0') when ps(0)= '1' else     (0 downto 0 => '0') & level0;
   level2<= level1 & (1 downto 0 => '0') when ps(1)= '1' else     (1 downto 0 => '0') & level1;
   level3<= level2 & (3 downto 0 => '0') when ps(2)= '1' else     (3 downto 0 => '0') & level2;
   level4<= level3 & (7 downto 0 => '0') when ps(3)= '1' else     (7 downto 0 => '0') & level3;
   level5<= level4 & (15 downto 0 => '0') when ps(4)= '1' else     (15 downto 0 => '0') & level4;
   level6<= level5 & (31 downto 0 => '0') when ps(5)= '1' else     (31 downto 0 => '0') & level5;
   level7<= level6 & (63 downto 0 => '0') when ps(6)= '1' else     (63 downto 0 => '0') & level6;
   R <= level7(150 downto 0);
end architecture;

--------------------------------------------------------------------------------
--                          IEEEFPFMA_8_23_comb_uid2
-- Inputs: this FMA computes A*B+C
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2009-2019)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: A B C negateAB negateC RndMode
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IEEEFPFMA_8_23_comb_uid2 is
    port (A : in  std_logic_vector(31 downto 0);
          B : in  std_logic_vector(31 downto 0);
          C : in  std_logic_vector(31 downto 0);
          negateAB : in  std_logic;
          negateC : in  std_logic;
          RndMode : in  std_logic_vector(1 downto 0);
          R : out  std_logic_vector(31 downto 0)   );
end entity;

architecture arch of IEEEFPFMA_8_23_comb_uid2 is
   component RightShifter24_by_max_76_comb_uid4 is
      port ( X : in  std_logic_vector(23 downto 0);
             S : in  std_logic_vector(6 downto 0);
             R : out  std_logic_vector(99 downto 0)   );
   end component;

   component LZC_52_comb_uid6 is
      port ( I : in  std_logic_vector(51 downto 0);
             O : out  std_logic_vector(5 downto 0)   );
   end component;

   component LeftShifter76_by_max_75_comb_uid8 is
      port ( X : in  std_logic_vector(75 downto 0);
             S : in  std_logic_vector(6 downto 0);
             R : out  std_logic_vector(150 downto 0)   );
   end component;

signal Asgn :  std_logic;
signal AexpField :  std_logic_vector(7 downto 0);
signal AsigField :  std_logic_vector(22 downto 0);
signal AisNormal :  std_logic;
signal AisInfOrNaN :  std_logic;
signal AhasNonNullSig :  std_logic;
signal AisZero :  std_logic;
signal AisInf :  std_logic;
signal AisNaN :  std_logic;
signal Bsgn :  std_logic;
signal BexpField :  std_logic_vector(7 downto 0);
signal BsigField :  std_logic_vector(22 downto 0);
signal BisNormal :  std_logic;
signal BisInfOrNaN :  std_logic;
signal BhasNonNullSig :  std_logic;
signal BisZero :  std_logic;
signal BisInf :  std_logic;
signal BisNaN :  std_logic;
signal Aexp :  std_logic_vector(7 downto 0);
signal Bexp :  std_logic_vector(7 downto 0);
signal Asig :  std_logic_vector(23 downto 0);
signal Bsig :  std_logic_vector(23 downto 0);
signal AexpPlusBexp :  std_logic_vector(8 downto 0);
signal Csgn :  std_logic;
signal CexpField :  std_logic_vector(7 downto 0);
signal CsigField :  std_logic_vector(22 downto 0);
signal CisNormal :  std_logic;
signal CisInfOrNaN :  std_logic;
signal ChasNonNullSig :  std_logic;
signal CisZero :  std_logic;
signal CisInf :  std_logic;
signal CisNaN :  std_logic;
signal RisNaN :  std_logic;
signal tentativeRisInf :  std_logic;
signal Cexp :  std_logic_vector(7 downto 0);
signal effectiveSub :  std_logic;
signal Csig :  std_logic_vector(23 downto 0);
signal expDiffPrepare :  std_logic_vector(9 downto 0);
signal expDiff :  std_logic_vector(9 downto 0);
signal tmpExpComp1 :  std_logic_vector(9 downto 0);
signal expDiffVerySmall :  std_logic;
signal tmpExpComp2 :  std_logic_vector(9 downto 0);
signal expDiffSmall :  std_logic;
signal tmpExpComp3 :  std_logic_vector(9 downto 0);
signal expDiffNotLarge :  std_logic;
signal ShiftValue :  std_logic_vector(6 downto 0);
signal CsigShifted :  std_logic_vector(99 downto 0);
signal sticky1 :  std_logic;
signal CsigShiftedT :  std_logic_vector(75 downto 0);
signal P :  std_logic_vector(47 downto 0);
signal Paligned :  std_logic_vector(75 downto 0);
signal CsigInverted :  std_logic_vector(76 downto 0);
signal BigSum :  std_logic_vector(76 downto 0);
signal BigSum2 :  std_logic_vector(76 downto 0);
signal RsgnTentative :  std_logic;
signal BigSumAbs :  std_logic_vector(75 downto 0);
signal BigSumAbsLowerBits :  std_logic_vector(51 downto 0);
signal L :  std_logic_vector(5 downto 0);
signal tmpExpCompRes1 :  std_logic_vector(9 downto 0);
signal tmpExpCompRes2 :  std_logic_vector(9 downto 0);
signal RisSubNormal :  std_logic;
signal RisZero :  std_logic;
signal RisSubNormalOrZero :  std_logic;
signal Rsgn :  std_logic;
signal shiftValueCaseSubnormal :  std_logic_vector(8 downto 0);
signal normShiftValue :  std_logic_vector(6 downto 0);
signal BigSumNormd :  std_logic_vector(150 downto 0);
signal expTentative :  std_logic_vector(9 downto 0);
signal sticky2 :  std_logic;
signal fracTentative :  std_logic_vector(26 downto 0);
signal fracLeadingBitsNormal :  std_logic_vector(1 downto 0);
signal fracLeadingBits :  std_logic_vector(1 downto 0);
signal fracResultNormd :  std_logic_vector(22 downto 0);
signal fracResultRoundBit :  std_logic;
signal fracResultStickyBit :  std_logic;
signal round :  std_logic;
signal expUpdate :  std_logic_vector(9 downto 0);
signal exponentResult1 :  std_logic_vector(9 downto 0);
signal resultBeforeRound :  std_logic_vector(32 downto 0);
signal resultRounded :  std_logic_vector(32 downto 0);
signal Roverflowed :  std_logic;
signal finalRisInf :  std_logic;
signal Inf :  std_logic_vector(30 downto 0);
signal NaN :  std_logic_vector(30 downto 0);
begin

    -- Input decomposition 
   Asgn <= A(31);
   AexpField <= A(30 downto 23);
   AsigField <= A(22 downto 0);
   AisNormal <= A(23) or A(24) or A(25) or A(26) or A(27) or A(28) or A(29) or A(30);
   AisInfOrNaN <= A(23) and A(24) and A(25) and A(26) and A(27) and A(28) and A(29) and A(30);
   AhasNonNullSig <= A(0) or A(1) or A(2) or A(3) or A(4) or A(5) or A(6) or A(7) or A(8) or A(9) or A(10) or A(11) or A(12) or A(13) or A(14) or A(15) or A(16) or A(17) or A(18) or A(19) or A(20) or A(21) or A(22);
   AisZero <= (not AisNormal) and not AhasNonNullSig;
   AisInf <= AisInfOrNaN and not AhasNonNullSig;
   AisNaN <= AisInfOrNaN and AhasNonNullSig;
   Bsgn <= B(31);
   BexpField <= B(30 downto 23);
   BsigField <= B(22 downto 0);
   BisNormal <= B(23) or B(24) or B(25) or B(26) or B(27) or B(28) or B(29) or B(30);
   BisInfOrNaN <= B(23) and B(24) and B(25) and B(26) and B(27) and B(28) and B(29) and B(30);
   BhasNonNullSig <= B(0) or B(1) or B(2) or B(3) or B(4) or B(5) or B(6) or B(7) or B(8) or B(9) or B(10) or B(11) or B(12) or B(13) or B(14) or B(15) or B(16) or B(17) or B(18) or B(19) or B(20) or B(21) or B(22);
   BisZero <= (not BisNormal) and not BhasNonNullSig;
   BisInf <= BisInfOrNaN and not BhasNonNullSig;
   BisNaN <= BisInfOrNaN and BhasNonNullSig;

   -- unbiased exponents make everything simpler but may lead to suboptimal arch 
   Aexp <= AexpField - ("0" & (6 downto 1 => '1') & AisNormal);
   Bexp <= BexpField - ("0" & (6 downto 1 => '1') & BisNormal);

   -- mantissa with the implicit 1 or 0 appended 
   Asig <= AisNormal & AsigField ;
   Bsig <= BisNormal & BsigField ;
   AexpPlusBexp <= (Aexp(7) & Aexp) + (Bexp(7) & Bexp) ;
   Csgn <= C(31);
   CexpField <= C(30 downto 23);
   CsigField <= C(22 downto 0);
   CisNormal <= C(23) or C(24) or C(25) or C(26) or C(27) or C(28) or C(29) or C(30);
   CisInfOrNaN <= C(23) and C(24) and C(25) and C(26) and C(27) and C(28) and C(29) and C(30);
   ChasNonNullSig <= C(0) or C(1) or C(2) or C(3) or C(4) or C(5) or C(6) or C(7) or C(8) or C(9) or C(10) or C(11) or C(12) or C(13) or C(14) or C(15) or C(16) or C(17) or C(18) or C(19) or C(20) or C(21) or C(22);
   CisZero <= (not CisNormal) and not ChasNonNullSig;
   CisInf <= CisInfOrNaN and not ChasNonNullSig;
   CisNaN <= CisInfOrNaN and ChasNonNullSig;
   -- result NaN iff one input is NaN, or 0*inf+x, or +/-(inf-inf)
   -- not tentative: the last case may not happen for finite A and B and inf C, as in this case AB remains finite
   RisNaN <= AisNaN or BisNaN or CisNaN or ((AisInf or BisInf) and CisInf and ((Asgn xor Bsgn) xor Csgn));
   -- result inf iff either AB or C is inf, and both have the same sign
   -- tentative, AB+C may overflow
   tentativeRisInf <= (((AisInf and not (BisZero or BisNaN)) or (BisInf and not (AisZero or AisNaN))) and (not CisNaN) and ((CisInf and not ((Asgn xor Bsgn) xor Csgn)) or not CisInf))
      or (CisInf and (not (AisNaN or BisNaN)) and (((AisInf or BisInf) and not ((Asgn xor Bsgn) xor Csgn)) or not (AisInf or BisInf)));

   -- unbiased exponents make everything simpler but may lead to suboptimal arch 
   Cexp <= CexpField - ("0" & (6 downto 1 => '1') & CisNormal);
   effectiveSub <= (negateAB xor Asgn xor Bsgn) xor (negateC xor Csgn);

   -- mantissa with the implicit 1 or 0 appended 
   Csig <= CisNormal & CsigField ;

    -- Computation of the exponent difference 
   expDiffPrepare <= ("00" & AexpField) + ("00" & BexpField) - ("000" & (6 downto 1 => '1') & AisNormal) - BisNormal;
   expDiff <= ("00" & CexpField) - expDiffPrepare - CisNormal;

    -- Addend shift datapath 
   -- Some logic to determine shift distance and tentative result exponent 
   tmpExpComp1 <= expDiff + "0000110000";
   expDiffVerySmall <= tmpExpComp1(9);  -- if expDiff < -2p
   tmpExpComp2 <= expDiff - "0000000011";
   expDiffSmall <= tmpExpComp2(9);  -- if expDiff < 3
   tmpExpComp3 <= expDiff - "0000011011";
   expDiffNotLarge <= tmpExpComp3(9);  -- if expDiff < p+3
   ShiftValue <= 
           "1001100" when expDiffVerySmall='1'
      else "0011011" - (expDiff (6 downto 0)) when expDiffNotLarge='1'
      else "0000000" ;
   RightShifterComponent: RightShifter24_by_max_76_comb_uid4
      port map ( S => ShiftValue,
                 X => Csig,
                 R => CsigShifted);
   sticky1 <= CsigShifted(0) or CsigShifted(1) or CsigShifted(2) or CsigShifted(3) or CsigShifted(4) or CsigShifted(5) or CsigShifted(6) or CsigShifted(7) or CsigShifted(8) or CsigShifted(9) or CsigShifted(10) or CsigShifted(11) or CsigShifted(12) or CsigShifted(13) or CsigShifted(14) or CsigShifted(15) or CsigShifted(16) or CsigShifted(17) or CsigShifted(18) or CsigShifted(19) or CsigShifted(20) or CsigShifted(21) or CsigShifted(22) or CsigShifted(23);
   CsigShiftedT <= CsigShifted(99 downto 24);

    -- Product datapath (using naive * operator, may improve in the future)
   P <= Asig * Bsig ;
   Paligned <= (25 downto 0 => '0') & P & "00";

    -- The sum at last 
   CsigInverted <= ('0' &CsigShiftedT) when effectiveSub='0'  else ('1' & not CsigShiftedT);
   BigSum <= CsigInverted + ('0' & Paligned) + effectiveSub;  -- P +/-CeffectiveSub is a carry in
   BigSum2 <= CsigShiftedT - ('0' & Paligned);
   RsgnTentative <= Asgn xor Bsgn xor negateAB xor BigSum(76);
   BigSumAbs <= BigSum(75 downto 0) when (BigSum2(76) or not effectiveSub)='1' else BigSum2(75 downto 0);
   BigSumAbsLowerBits <= BigSumAbs(51 downto 0);
   IEEEFPFMA_8_23_comb_uid2LeadingZeroCounter: LZC_52_comb_uid6
      port map ( I => BigSumAbsLowerBits,
                 O => L);
   tmpExpCompRes1 <= (AexpPlusBexp(8) & AexpPlusBexp) + "0010000001";
   tmpExpCompRes2 <= tmpExpCompRes1 - ((9 downto 6 => '0') & L);
   RisSubNormal <= (expDiffSmall or not CisNormal) and  tmpExpCompRes2(9);
   RisZero <= expDiffSmall when L="110100"  else '0';
   RisSubNormalOrZero <= RisSubNormal or RisZero;
   Rsgn <=      (Asgn xor Bsgn xor negateAB) and (Csgn xor negateC) when ((AisZero or BisZero) and CisZero)='1'  -- negative only for -0 + -0 
      else '0' when RisZero='1'  -- covers 1-1 = +0
      else RsgnTentative;  -- covers to underflow to zero case
   shiftValueCaseSubnormal <= AexpPlusBexp + "010011001";
   normShiftValue <= 
           L + "0011001" when (expDiffSmall and not RisSubNormal)='1'
      else shiftValueCaseSubnormal(6 downto 0) when (expDiffSmall and RisSubNormal)='1'
      else ShiftValue; -- undo inital shift
   NormalizationShifter: LeftShifter76_by_max_75_comb_uid8
      port map ( S => normShiftValue,
                 X => BigSumAbs,
                 R => BigSumNormd);
   -- TODO opt: push all the constant additions to exponentUpdate
   expTentative <= 
           "1110000011" when RisZero='1'
      else "1110000010" when RisSubNormal='1'
      else (AexpPlusBexp(8) & AexpPlusBexp - ((9 downto 6 => '0') & L))  + "0000000011" when (expDiffSmall and not RisSubNormal)='1'
      else (Cexp(7) & Cexp(7) & Cexp) + "0000000001" ;
   sticky2 <= BigSumNormd(0) or BigSumNormd(1) or BigSumNormd(2) or BigSumNormd(3) or BigSumNormd(4) or BigSumNormd(5) or BigSumNormd(6) or BigSumNormd(7) or BigSumNormd(8) or BigSumNormd(9) or BigSumNormd(10) or BigSumNormd(11) or BigSumNormd(12) or BigSumNormd(13) or BigSumNormd(14) or BigSumNormd(15) or BigSumNormd(16) or BigSumNormd(17) or BigSumNormd(18) or BigSumNormd(19) or BigSumNormd(20) or BigSumNormd(21) or BigSumNormd(22) or BigSumNormd(23) or BigSumNormd(24) or BigSumNormd(25) or BigSumNormd(26) or BigSumNormd(27) or BigSumNormd(28) or BigSumNormd(29) or BigSumNormd(30) or BigSumNormd(31) or BigSumNormd(32) or BigSumNormd(33) or BigSumNormd(34) or BigSumNormd(35) or BigSumNormd(36) or BigSumNormd(37) or BigSumNormd(38) or BigSumNormd(39) or BigSumNormd(40) or BigSumNormd(41) or BigSumNormd(42) or BigSumNormd(43) or BigSumNormd(44) or BigSumNormd(45) or BigSumNormd(46) or BigSumNormd(47) or BigSumNormd(48) or BigSumNormd(49);

   fracTentative <= BigSumNormd(76 downto 50);

    -- Last 2-bit normalization 
   fracLeadingBitsNormal <=  fracTentative(26 downto 25) ;
   fracLeadingBits <= "01" when RisSubNormal='1' else  fracLeadingBitsNormal;
   fracResultNormd <=
           fracTentative(23 downto 1)  when fracLeadingBits = "00" 
      else fracTentative(24 downto 2)  when fracLeadingBits = "01" 
      else fracTentative(25 downto 3);
   fracResultRoundBit <=
           fracTentative(0) 	 when fracLeadingBits = "00" 
      else fracTentative(1)    when fracLeadingBits = "01" 
      else fracTentative(2) ;
   fracResultStickyBit <=
           sticky1 or sticky2	 when fracLeadingBits = "00" 
      else fracTentative(0) or sticky1 or sticky2    when fracLeadingBits = "01" 
      else fracTentative(1) or fracTentative(0) or  sticky1 or sticky2;
   round <= fracResultRoundBit and (fracResultStickyBit or fracResultNormd(0));
   expUpdate <= "0001111101" when RisZero = '1'       -- bias - 2
         else   "0001111101" when fracLeadingBits = "00" -- bias - 2
         else   "0001111110" when fracLeadingBits = "01" -- bias - 1 
         else   "0001111111";                            -- bias 
   exponentResult1 <= expTentative + expUpdate;
   resultBeforeRound <= exponentResult1 & fracResultNormd;
   resultRounded <= resultBeforeRound + ((32 downto 1 => '0') & round);
   Roverflowed <= resultRounded(32) or resultRounded(31) or (resultRounded(30) and resultRounded(29) and resultRounded(28) and resultRounded(27) and resultRounded(26) and resultRounded(25) and resultRounded(24) and resultRounded(23));
   finalRisInf <= tentativeRisInf or Roverflowed; 
   Inf <= (30 downto 23 => '1') & (22 downto 0 => '0');
   NaN <= (30 downto 23 => '1') & (22 downto 0 => '1');
   R <= 
           Rsgn & Inf when ((not RisNaN) and finalRisInf)='1'
      else '0'  & NaN when RisNaN='1'
      else Rsgn & resultRounded(30 downto 0);
end architecture;



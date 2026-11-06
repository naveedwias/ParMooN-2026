#include "Enumerations_fe.h"
#include "templateNames.h"
#include "MooNMD_Io.h"

std::ostream& operator<<(std::ostream& out, const BFRefElements t)
{
  switch(t)
  {
    case BFRefElements::BFUnitLine: out << "BFUnitLine"; break;
    case BFRefElements::BFUnitTriangle: out << "BFUnitTriangle"; break;
    case BFRefElements::BFUnitSquare: out << "BFUnitSquare"; break;
    case BFRefElements::BFUnitTetrahedron: out << "BFUnitTetrahedron"; break;
    case BFRefElements::BFUnitHexahedron: out << "BFUnitHexahedron"; break;
    default: out << "unknown reference element"; break;
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const HNDesc t)
{
  const char* s = 0;
#define PROCESS_VAL(p)                                                         \
  case(HNDesc::p):                                                             \
    s = #p;                                                                    \
    break;
  switch(t)
  {
    PROCESS_VAL(HN_C_P1_2D_0);
    PROCESS_VAL(HN_C_P2_2D_0);
    PROCESS_VAL(HN_C_P2_2D_1);
    PROCESS_VAL(HN_C_P3_2D_0);
    PROCESS_VAL(HN_C_P3_2D_1);
    PROCESS_VAL(HN_C_P3_2D_2);
    PROCESS_VAL(HN_C_P4_2D_0);
    PROCESS_VAL(HN_C_P4_2D_1);
    PROCESS_VAL(HN_C_P4_2D_2);
    PROCESS_VAL(HN_C_P4_2D_3);
    PROCESS_VAL(HN_C_P5_2D_0);
    PROCESS_VAL(HN_C_P5_2D_1);
    PROCESS_VAL(HN_C_P5_2D_2);
    PROCESS_VAL(HN_C_P5_2D_3);
    PROCESS_VAL(HN_C_P5_2D_4);
    PROCESS_VAL(HN_C_P6_2D_0);
    PROCESS_VAL(HN_C_P6_2D_1);
    PROCESS_VAL(HN_C_P6_2D_2);
    PROCESS_VAL(HN_C_P6_2D_3);
    PROCESS_VAL(HN_C_P6_2D_4);
    PROCESS_VAL(HN_C_P6_2D_5);
    PROCESS_VAL(HN_C_P7_2D_0);
    PROCESS_VAL(HN_C_P7_2D_1);
    PROCESS_VAL(HN_C_P7_2D_2);
    PROCESS_VAL(HN_C_P7_2D_3);
    PROCESS_VAL(HN_C_P7_2D_4);
    PROCESS_VAL(HN_C_P7_2D_5);
    PROCESS_VAL(HN_C_P7_2D_6);
    PROCESS_VAL(HN_C_P8_2D_0);
    PROCESS_VAL(HN_C_P8_2D_1);
    PROCESS_VAL(HN_C_P8_2D_2);
    PROCESS_VAL(HN_C_P8_2D_3);
    PROCESS_VAL(HN_C_P8_2D_4);
    PROCESS_VAL(HN_C_P8_2D_5);
    PROCESS_VAL(HN_C_P8_2D_6);
    PROCESS_VAL(HN_C_P8_2D_7);
    PROCESS_VAL(HN_C_P9_2D_0);
    PROCESS_VAL(HN_C_P9_2D_1);
    PROCESS_VAL(HN_C_P9_2D_2);
    PROCESS_VAL(HN_C_P9_2D_3);
    PROCESS_VAL(HN_C_P9_2D_4);
    PROCESS_VAL(HN_C_P9_2D_5);
    PROCESS_VAL(HN_C_P9_2D_6);
    PROCESS_VAL(HN_C_P9_2D_7);
    PROCESS_VAL(HN_C_P9_2D_8);
    PROCESS_VAL(HN_N_P1_2D_0);
    PROCESS_VAL(HN_N_P2_2D_0);
    PROCESS_VAL(HN_N_P3_2D_0);
    PROCESS_VAL(HN_N_P4_2D_0);
    PROCESS_VAL(HN_N_P5_2D_0);
    PROCESS_VAL(HN_C_Q1_3D_E);
    PROCESS_VAL(HN_C_Q1_3D_F);
    PROCESS_VAL(HN_C_Q2_3D_E);
    PROCESS_VAL(HN_C_Q2_3D_F);
    PROCESS_VAL(HN_C_Q3_3D_1);
    PROCESS_VAL(HN_C_Q3_3D_2);
    PROCESS_VAL(HN_C_Q3_3D_3);
    PROCESS_VAL(HN_C_Q3_3D_4);
    PROCESS_VAL(HN_C_Q3_3D_5);
    PROCESS_VAL(HN_C_P1_3D_E);
    PROCESS_VAL(HN_C_P2_3D_E);
    PROCESS_VAL(HN_C_P2_3D_F);
    PROCESS_VAL(HN_C_P3_3D_E);
    PROCESS_VAL(HN_C_P3_3D_M);
    PROCESS_VAL(HN_C_P3_3D_F);
    PROCESS_VAL(HN_C_P3_3D_G);
    PROCESS_VAL(HN_N_P1_3D_E);
    PROCESS_VAL(HN_N_P2_3D_0);
    PROCESS_VAL(HN_N_P2_3D_1);
    PROCESS_VAL(HN_N_P2_3D_2);
    default:
      s = "unknown HNDesc type";
      break;
  }
#undef PROCESS_VAL
  return out << s;
}

std::ostream& operator<<(std::ostream& out, const MultiIndex1D i)
{
  switch (i)
  {
    case MultiIndex1D::D0: out << "D0"; break;
    case MultiIndex1D::D1: out << "D1"; break;
    case MultiIndex1D::D2: out << "D2"; break;
    default: out << "???"; break;
  }

  return out;
}

std::ostream& operator<<(std::ostream& out, const MultiIndex2D i)
{
  switch (i)
  {
    case MultiIndex2D::D00: out << "D00"; break;
    case MultiIndex2D::D10: out << "D10"; break;
    case MultiIndex2D::D01: out << "D01"; break;
    case MultiIndex2D::D20: out << "D20"; break;
    case MultiIndex2D::D11: out << "D11"; break;
    case MultiIndex2D::D02: out << "D02"; break;
    default: out << "???"; break;
  }

  return out;
}

std::ostream& operator<<(std::ostream& out, const MultiIndex3D i)
{
  switch (i)
  {
    case MultiIndex3D::D000: out << "D000"; break;
    case MultiIndex3D::D100: out << "D100"; break;
    case MultiIndex3D::D010: out << "D010"; break;
    case MultiIndex3D::D001: out << "D001"; break;
    case MultiIndex3D::D200: out << "D200"; break;
    case MultiIndex3D::D110: out << "D110"; break;
    case MultiIndex3D::D101: out << "D101"; break;
    case MultiIndex3D::D020: out << "D020"; break;
    case MultiIndex3D::D011: out << "D011"; break;
    case MultiIndex3D::D002: out << "D002"; break;
    default: out << "???"; break;
  }

  return out;
}

template<>
Template_names<2>::MultiIndex_vector indices_up_to_order<2>(int order)
{
  switch(order)
  {
    case 0:
      return { MultiIndex2D::D00 };
      break;
    case 1:
      return { MultiIndex2D::D00, MultiIndex2D::D10, MultiIndex2D::D01 };
      break;
    case 2:
      return { MultiIndex2D::D00, MultiIndex2D::D10, MultiIndex2D::D01,
               MultiIndex2D::D20, MultiIndex2D::D11, MultiIndex2D::D02 };
      break;
    default:
      ErrThrow("Multi-indices only available for orders 0,1, and 2");
      break;
  }
}

template<>
Template_names<3>::MultiIndex_vector indices_up_to_order<3>(int order)
{
  switch(order)
  {
    case 0:
      return { MultiIndex3D::D000 };
      break;
    case 1:
      return { MultiIndex3D::D000, MultiIndex3D::D100, MultiIndex3D::D010,
               MultiIndex3D::D001 };
      break;
    case 2:
      return { MultiIndex3D::D000,
               MultiIndex3D::D100, MultiIndex3D::D010, MultiIndex3D::D001,
               MultiIndex3D::D200, MultiIndex3D::D110, MultiIndex3D::D101,
               MultiIndex3D::D020, MultiIndex3D::D011, MultiIndex3D::D002 };
      break;
    default:
      ErrThrow("Multi-indices only available for orders 0,1, and 2");
      break;
  }
}

template<>
Template_names<2>::MultiIndex_vector indices_of_order<2>(int order)
{
  switch(order)
  {
    case 0:
      return { MultiIndex2D::D00 };
      break;
    case 1:
      return { MultiIndex2D::D10, MultiIndex2D::D01 };
      break;
    case 2:
      return { MultiIndex2D::D20, MultiIndex2D::D11, MultiIndex2D::D02 };
      break;
    default:
      ErrThrow("Multi-indices only available for orders 0,1, and 2");
      break;
  }
}

template<>
Template_names<3>::MultiIndex_vector indices_of_order<3>(int order)
{
  switch(order)
  {
    case 0:
      return { MultiIndex3D::D000 };
      break;
    case 1:
      return { MultiIndex3D::D100, MultiIndex3D::D010, MultiIndex3D::D001 };
      break;
    case 2:
      return { MultiIndex3D::D200, MultiIndex3D::D110, MultiIndex3D::D101,
               MultiIndex3D::D011, MultiIndex3D::D020, MultiIndex3D::D002 };
      break;
    default:
      ErrThrow("Multi-indices only available for orders 0,1, and 2");
      break;
  }
}

template<>
Template_names<2>::MultiIndex_vector laplacian_indices<2>()
{
  return { MultiIndex2D::D20, MultiIndex2D::D02 };
}

template<>
Template_names<3>::MultiIndex_vector laplacian_indices<3>()
{
  return { MultiIndex3D::D200, MultiIndex3D::D020, MultiIndex3D::D002 };
}

template<>
bool needs_second_derivatives<2>(const Template_names<2>::MultiIndex_vector& indices)
{
  for (auto i: indices)
  {
    switch(i)
    {
      case MultiIndex2D::D20:
      case MultiIndex2D::D11:
      case MultiIndex2D::D02:
        return true;

      default:
        continue;
    }
  }

  return false;
}

template<>
bool needs_second_derivatives<3>(const Template_names<3>::MultiIndex_vector& indices)
{
  for (auto i: indices)
  {
    switch(i)
    {
      case MultiIndex3D::D200:
      case MultiIndex3D::D020:
      case MultiIndex3D::D002:
      case MultiIndex3D::D110:
      case MultiIndex3D::D101:
      case MultiIndex3D::D011:
        return true;

      default:
        continue;
    }
  }

  return false;
}
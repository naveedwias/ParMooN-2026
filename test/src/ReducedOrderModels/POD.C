/** **************************************************************************** 
*
* @name       POD
* @brief      Computation/manipulation of POD basis
*             independently of problem dimension
*
* @author     Swetlana Giere
* @date       08.03.2017 (start of implementation)
*
*******************************************************************************/

#include <algorithm>
#include <cmath>
#include <POD.h>
#include <SnapshotsCollector.h>
#include <Solver.h>

#ifdef _MPI
#include <mpi.h>
#endif

// Extern declaration of Lapack method.
extern "C" {
//  computes the eigenvalues and eigenvectors
void dsyev_(char* jobz, char* uplo, int* n, double* a, int* lda, double* w,
            double* work, int* lwork, int* info);
}

/** ***************************************************************************/
ParameterDatabase POD::default_pod_database()
{
  ParameterDatabase db("Default ParMooN parameter database for "
                       "POD-basis computation");

  db.add("pod_directory", ".",
         "This directory is where the POD basis and Co. are written. This "
         "directory will be created, if it does not exist already. Files in "
         "this directory will be overwritten without any warning.");

  db.add("pod_basename", "parmoon_pod",
         "Basename for pod basis and related files. When writing the POD "
         "basis, the basis elements will be written into pod_basename.pod, "
         "the average (if needed) into pod_basename.mean. When reading the "
         "basis, the program expect to find files ending with .pod and .mean");

  db.add("pod_binary", false,
         "Write the POD basis/mean as text or binary?",
         {true,false});

  db.add("pod_rank", (size_t) 0,
         "This integer specifies the dimension of the POD space to be computed."
         " If pod_rank <= 0, then all possible POD modes will be computed.");

  db.add("pod_fluctuations_only", true,
         "This is the flag whether the POD basis should be computed only "
         "fluctuation part (without average) of the snapshots "
         "(also central trajectory method).",
         {true,false});

  db.add("pod_inner_product", "euclidean",
         "Specification of the inner product which is used to compute POD "
         "basis. Besides default value, 'L2' (and 'H1' for Navier Stokes) are "
         "possible at the moment.",
         {"euclidean", "L2", "H1"});

  db.add("pod_dt_projection_inner_product", "L2",
         "Specification of the inner product which is used to project time "
         "derivative snapshots back into the FE space.",
         {"euclidean", "L2", "H1"});

  db.add("pod_dt_projection_div_fix", false,
         "Should the projected time derivatives be then projected onto the "
         "discretely divergence-free subspace?",
         { true, false });

  db.add("pod_dt_projection_average", false,
         "Should the projected time derivatives have the average's time "
         "derivative subtracted?",
         { true, false });

  db.add("pod_dt_projection_regularized", false,
         "Should the projected time derivatives be regularized?",
         { true, false });

  db.add("pod_dt_projection_filter_width", 1.0,
         "Filter width for the differential filter (Helmholtz equation) for "
         "the computation of the regularized time derivatives.",
         0., 10.);

  db.add("pod_dt_projection_filter_local_scale", false,
         "Whether to apply local scaling to the differential filter when "
         "computing the regularized time derivatives.",
         { true, false });

  // Merge with snapshots database needed to get the path of the snapshots
  db.merge(SnapshotsCollector::default_snapshots_database(), true);

  return db;
}

/** ***************************************************************************/
POD::POD(const ParameterDatabase& param_db, std::string filename_suffix) :
  db(POD::default_pod_database()),
  rank(0),
  length(0),
  length_snaps(0),
  number_snaps(0),
  eigen_threshold(1.0e-10),
  valid_eigs(0),
  eigs(nullptr),
  gramian_ptr(nullptr),
  projection_solver_db(Solver<>::default_solver_database()),
  snaps_mat(0, 0),
  snaps_mean(0),
  pod_basis(0, 0)
{
  this->db.merge(param_db, false);
  POD::check_parameters();
  rank = db["pod_rank"];

  set_basename(filename_suffix);
}

/** ***************************************************************************/
void POD::set_basename(std::string filename_suffix)
{
  this->filename_suffix = filename_suffix;
  this->basename =  this->db["pod_directory"].get<std::string>() + "/"
                  + this->db["pod_basename"].get<std::string>() + ".";

  if (!filename_suffix.empty())
  {
    this->basename += filename_suffix + ".";
  }

  this->basename += this->db["pod_inner_product"].get<std::string>() + ".";

  if (this->db["pod_fluctuations_only"])
  {
    basename += "fluc.";
  }
}

/** ***************************************************************************/
POD::~POD()
{
  if (eigs != nullptr)
  {
    delete[] eigs;
  }
}

/** ***************************************************************************/
void POD::check_parameters()
{
  if (!this->db["pod_inner_product"].is("L2")
    && !this->db["pod_inner_product"].is("H1")
    && !this->db["pod_inner_product"].is("euclidean"))
  {
    ErrThrow("Error: POD::check_parameters(), the parameter ",
             "'pod_inner_product' does not currently support the value ",
             this->db["pod_inner_product"].get<std::string>(), ".");
  }
}

/** ***************************************************************************/
void POD::set_gramian_ptr(std::shared_ptr<TMatrix> mat)
{
  this->gramian_ptr = mat;
}

void POD::set_projection(const BlockFEMatrix& mat, const ParameterDatabase& solver_db)
{
  projection_matrix = mat;
  projection_solver_db.merge(solver_db, false);
}

/** ***************************************************************************/
void POD::read_snapshots()
{
  if (length <= 0)
  {
    ErrThrow("POD read_snapshots(): length is not available!");
  }

  // read snapshots from file
  bool bin = false;

  std::string snap_filename = SnapshotsCollector::get_snapshot_filename(this->db, this->filename_suffix);

  if (db["snaps_binary"])
  {
    bin = true;
  }

  Output::root_info<1>("POD", "Reading snapshots from file: ", snap_filename, "...");

  std::vector < std::vector<double> > tmp_snaps;
  SnapshotsCollector::read_data(snap_filename, bin, length, tmp_snaps);

  this->length_snaps = tmp_snaps[0].size(); //total dof
  this->number_snaps = tmp_snaps.size();

  // resize snaps_mat
  DenseMatrix snaps_mat(this->number_snaps, this->length_snaps);
  this->snaps_mat = snaps_mat;

  // store snapshots into member matrix snaps_mat 
  for (int i = 0; i < this->length_snaps; i++)
  {
    for (int j = 0; j < this->number_snaps; j++)
    {
      this->snaps_mat.setEntry(j, i, tmp_snaps[j][i]);
    }
  }

  Output::root_info<1>("POD", "... snapshots read.");
  Output::root_info<1>("POD", "  * Length of snapshots : ", this->length_snaps);
  Output::root_info<1>("POD", "  * Number of snapshots : ", this->number_snaps);
}

void POD::compute_snaps_average(double* mean, int length)
{
  this->length = length;
  if (this->length_snaps == 0)
  {
    read_snapshots();
  }

  bool with_time_derivatives = db["snaps_time_derivative"];

  if (with_time_derivatives)
  {
    Output::root_info("POD", "Snapshots include time derivatives - averaging every second snapshot only.");
  }

  double m = 1.0 / (with_time_derivatives ? ((this->number_snaps - 1) / 2 + 1) : this->number_snaps);

  for (int i = 0; i < length_snaps; i++)
  {
    // calculate sum of the ith column of snapshot matrix
    double sum_i = 0.0;

    if (with_time_derivatives)
    {
      // no time derivatives are written for the first entry

      sum_i = snaps_mat.getEntry(0, i);

      for (int j = 1; j < this->number_snaps; j += 2)
      {
        sum_i += snaps_mat.getEntry(j, i);
      }

      mean[i] = m * sum_i;
    }
    else
    {
      for (int j = 0; j < this->number_snaps; j++)
      {
        sum_i += snaps_mat.getEntry(j, i);
      }

      mean[i] = m * sum_i;
    }
  }
}

/** ***************************************************************************/
void POD::decompose_snaps()
{
  // transforms raw snaps into mean + fluctuations.
  // this is allowed to be MPI-agnostic because there's no interaction

  this->snaps_mean.resize(this->length_snaps);

  if (this->length_snaps == 0)
  {
    ErrThrow("Error: Snapshots are not available. Before computing POD basis "
             "read_snapshots() has to be called!");
  }

  bool with_time_derivatives = db["snaps_time_derivative"];

  compute_snaps_average(snaps_mean.data(), length_snaps);

  if (with_time_derivatives)
  {
    snaps_mat.addToRow(&snaps_mean, 0, -1.0);

    for (int j = 1; j < this->number_snaps; j += 2)
    {
      snaps_mat.addToRow(&snaps_mean, j, -1.0);
    }
  }
  else
  {
    // subtract the mean from snapshots
    bool transpose = true;
    snaps_mat.add(&snaps_mean, -1.0, transpose);
  }
}

void POD::regularize_time_derivatives()
{
  Output::root_info("POD", "Regularizing time derivatives...");

  if (!dt_regularizer)
  {
    ErrThrow("Time derivative regularizer is not set!");
  }

  double tau = db["snaps_time_derivative_tau"];

  if (tau == 0.0)
  {
    Output::root_info("POD", "...but actually, tau = 0, so no need.");
    return;
  }

  std::vector<double> dt(length);

  for (int j = 2; j < this->number_snaps; j += 2)
  {
    Output::root_info("POD", "Regularizing time derivative snapshot ", j / 2, "/", this->number_snaps / 2, "...");

    snaps_mat.getRow(j, dt.data());

    dt_regularizer(dt.data());

    snaps_mat.setRow(j, dt.data());
  }
}

/** ***************************************************************************/
void POD::compute_basis(int l)
{
  bool i_am_root = true;

#ifdef _MPI
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  i_am_root = mpi_rank == 0;
#endif

  this->length = l;
  read_snapshots();

  std::string name = "";
  if (!filename_suffix.empty())
  {
    name += filename_suffix + "-";
  }

  if (db["pod_fluctuations_only"])
  {
    Output::root_info<1>("POD", name, "POD will be computed from fluctuating part of "
                     "snapshots...");
    POD::decompose_snaps();
  }
  else
  {
    Output::root_info<1>("POD", name, "POD will be computed from full/raw snapshots...");
  }

  if (db["snaps_time_derivative"])
  {
    double tau = db["snaps_time_derivative_tau"];

    if (db["pod_dt_projection_average"] && db["pod_fluctuations_only"] && tau > 0.0)
    {
      Output::root_info("POD", "Getting average's time derivative");

      if (!dt_getter)
      {
        ErrThrow("Couldn't retrieve time derivative!");
      }

      std::vector<double> dt_mean(length_snaps);

      dt_getter(snaps_mean.data(), dt_mean.data());

      Output::root_info("POD", "Subtracting average from time derivatives");

      for (int j = 2; j < this->number_snaps; j += 2)
      {
        snaps_mat.addToRow(&dt_mean, j, -1.0);
      }
    }

    Output::root_info("POD", "Scaling time derivatives by tau = ", tau);

    for (int j = 2; j < this->number_snaps; j += 2)
    {
      snaps_mat.scaleRow(j, tau);
    }

    if (db["snaps_time_derivative_projection"])
    {
      Output::root_info("POD", "Projecting time derivatives back to FE "
        "space...");

      if (tau != 0.0)
      {
        auto solver = Solver<BlockFEMatrix, BlockVector>(projection_solver_db);
        solver.update_matrix(projection_matrix);

        BlockVector dt_u_star(projection_matrix, true);
        BlockVector dt_u(projection_matrix, false);

        for (int j = 2; j < this->number_snaps; j += 2)
        {
          Output::root_info("POD", "Projecting time derivative snapshot ", j / 2, "/", this->number_snaps / 2, "...");

          snaps_mat.getRow(j, dt_u_star.get_entries());
          dt_u_star.ResetNonActive();

          solver.solve(dt_u_star, dt_u);

      #ifdef _MPI
          // TODO: sync
      #endif

          snaps_mat.setRow(j, dt_u.get_entries());
        }

        if (db["pod_dt_projection_regularized"] && db["pod_dt_projection_filter_width"].get<double>() > 0.0)
        {
          regularize_time_derivatives();
        }

        if (db["pod_dt_projection_div_fix"])
        {
          if (!dt_div_fix)
          {
            ErrThrow("Time derivative divergence correction is not set!");
          }

          std::vector<double> dt(length);

          for (int j = 2; j < this->number_snaps; j += 2)
          {
            Output::root_info("POD", "Correcting time derivative divergence ", j / 2, "/", this->number_snaps / 2, "...");

            snaps_mat.getRow(j, dt.data());

            dt_div_fix(dt.data());

            snaps_mat.setRow(j, dt.data());
          }
        }
      }
      else
      {
        Output::root_info("POD", "...but actually, tau = 0, so no need.");
      }
    }
  }

  DenseMatrix autocorr_mat(this->number_snaps, this->number_snaps);
  compute_autocorr_mat(autocorr_mat);

  //----------------------------------------------------------------------------
  // compute eigen-values and eigen-vectors with LAPACK
  //----------------------------------------------------------------------------
  /* parameters for LAPACK dsyev */
  double** autocorr_to_eig;

  if (this->eigs != nullptr)
  {
    delete[] this->eigs;
  }

  /* array for eigenvalues */
  eigs = new double[this->number_snaps];

  autocorr_to_eig = new double*[this->number_snaps];
  autocorr_to_eig[0] = new double[this->number_snaps * number_snaps];

  for (int i = 0; i < this->number_snaps; i++)
  {
    autocorr_to_eig[i] = autocorr_to_eig[0] + i * this->number_snaps;
  }

  for (int i = 0; i < this->number_snaps; i++)
  {
    for (int j = 0; j < this->number_snaps; j++)
    {
      autocorr_to_eig[j][i] = autocorr_mat.getEntry(j, i);
    }
  }

  if (i_am_root)
  {
    // eigenvalue decomposition happens on root node only

    int n = (int)this->number_snaps;
    double* work;
    double wkopt;
    char arg1 = 'V'; /* compute eigenvalues and eigenvectors */
    char arg2 = 'U'; /* for upper triangular matrix */
    int info = 0, lwork;

    /* workspace query: the routine only calculates the optimal size of work */
    lwork = -1;
    dsyev_( &arg1, &arg2, &n, *autocorr_to_eig, &n, eigs, &wkopt, &lwork, &info );

    lwork = (int)wkopt;
    work = (double*)malloc(lwork * sizeof(double));

    /* Solve eigenproblem, the eigenvalues are stored in ascending order */
    dsyev_( &arg1, &arg2, &n, *autocorr_to_eig, &n, eigs, work, &lwork, &info );

    /* Check for convergence */
    if (info > 0)
    {
      ErrThrow("The algorithm failed to compute eigenvalues.");
    }

    //----------------------------------------------------------------------------
    // check eigen-values and rank
    //----------------------------------------------------------------------------
    /* count valid eigenvalues */
    for (int i = this->number_snaps - 1; i >= 0; i--)
    {
      if (this->eigs[i] > this->eigen_threshold)
      {
        this->valid_eigs++;
      }
      else
      {
        break;
      }
    }

    delete[] work;
  }

#ifdef _MPI
  // distribute eigenvalues
  MPI_Bcast(&(this->eigs), this->number_snaps, MPI_DOUBLE,
    0, MPI_COMM_WORLD);

  // distribute eigenvalue count
  MPI_Bcast(&(this->valid_eigs), 1, MPI_INT,
    0, MPI_COMM_WORLD);
#endif

  /* check the rank of the pod basis */
  if (this->rank <= 0)
  {
    this->rank = valid_eigs; // take all
  }

  if (this->rank > this->valid_eigs)
  {
    Output::root_warn<1>("POD basis", "Rank of POD basis can't be greater than ",
                    this->valid_eigs, "! Parameter 'pod_rank' changed from ",
                    this->rank, " to ", this->valid_eigs, ".");
    this->rank = this->valid_eigs;
  }

  Output::root_info<1>("POD", "  Number of valid ", name, "POD eigenvalues: ",
                   this->valid_eigs);
  Output::root_info<1>("POD", "  Dimension of ", name, "POD basis         : ", this->rank);

  /* Compute the sum of the resting eigenvalues: SUM_i=rank^valid_eigs eigs[i]
   * eigen-values are in ascending order */
  if (rank < valid_eigs)
  {
    double sum_rest_eigs = 0.;
    int nb_rest = std::max(0, valid_eigs - rank);
    for (int i = 0; i < nb_rest; i++)
    {
      int idx = this->number_snaps - 1 - rank - i;
      sum_rest_eigs += eigs[idx];
    }

    Output::root_info<1>("POD", "  Sum of the resting ", name, "eigenvalues : ",
                     sum_rest_eigs);
  }

  //----------------------------------------------------------------------------
  // compute pod_basis
  //----------------------------------------------------------------------------

  /* resize pod_basis */
  DenseMatrix pod_basis(this->rank, this->length_snaps);
  this->pod_basis = pod_basis;

  std::vector<double> vec_eig(this->number_snaps);
  std::vector<double> vec_Seig(this->length_snaps);

  // this is allowed to be MPI-agnostic again, since we're
  // assembling linear combinations of snapshots

  for (int i = 0; i < this->rank; i++)
  {
    /* eigen-values and eigen-vectors are in ascending order */
    int idx = this->number_snaps - 1 - i;

    for (int j = 0; j < this->number_snaps; j++)
    {
      vec_eig[j] = 1. / std::sqrt(this->number_snaps * this->eigs[idx])
                   * autocorr_to_eig[idx][j];
    }

    bool transpose = true;
    vec_Seig = this->snaps_mat.multiply(&vec_eig, 1., transpose);

    for (int j = 0; j < this->length_snaps; j++)
    {
      this->pod_basis.setEntry(i, j, vec_Seig[j]);
    }
  }

  /* reverse order of eigs (descending order) */
  std::reverse(this->eigs, this->eigs + this->number_snaps);

  this->length = this->length_snaps;
  delete[] *autocorr_to_eig;
  delete[] autocorr_to_eig;

  write_pod();
}

void write_data(std::string filename, const double* data, int columns, int rows, bool bin)
{
  if (bin)
  {
    std::ofstream ofile;
    ofile.open(filename.c_str(), std::ios::binary | std::ios::trunc | std::ios::out);

    if (!ofile.is_open())
    {
      ErrThrow("Error: File ", filename, " could not be created." );
    }

    ofile.write((char*)&columns, sizeof(int));
    // ofile.write((char*)data, sizeof(double) * columns * rows);

    std::vector<double> row(columns);
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < columns; j++)
      {
        row[j] = data[i + rows * j];
      }

      ofile.write((char*)(row.data()), sizeof(double) * columns);
    }

    ofile.close();
  }
  else
  {
    std::ofstream ofile;
    ofile.open(filename.c_str(), std::ios::out | std::ios::trunc );

    if (!ofile.is_open())
    {
      ErrThrow("Error: File ", filename, " could not be created." );
    }

    ofile << setprecision(12);

    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < columns; j++)
      {
        ofile << data[i + rows * j] << " ";
      }
      ofile << "\n";
    }

    ofile.close();
  }
}

void read_data_ex(std::string filename, std::vector<std::vector<double>> &data, int &columns, bool bin)
{
  data.clear();

  if (bin)
  {
    std::ifstream ifile(filename.c_str(), std::ios::binary);

    if (!ifile)
    {
      ErrThrow("File ", filename ," could not be opened. "
               "Check if the parameters in the input file are consistent "
               "with the parameters used to compute the POD basis.");
    }

    ifile.read((char*)&columns, sizeof(int));
    Output::info("POD", "Input file has ", columns, " columns.");

    unsigned int n = (unsigned)columns * sizeof(double);

    int r = 0;

    do
    {
      ++r;

      data.emplace_back(columns);

      ifile.read((char*)(data.back().data()), n);

      if (!ifile || ifile.gcount() < n)
      {
        if (ifile.gcount() > 0)
        {
          Output::root_warn("POD", "Row ", r, " has fewer than ",
            columns, " entries! Discarding.");
        }

        data.pop_back();
        break;
      }
    }
    while (true);

    ifile.close();
  }
  else
  {
    std::ifstream ifile(filename.c_str());

    if (!ifile)
    {
      ErrThrow("File ", filename ," could not be opened. "
               "Check if the parameters in the input file are consistent "
               "with the parameters used to compute the POD basis.");
    }

    std::string line;
    int r = 0;

    while (getline(ifile, line))
    {
      ++r;

      data.emplace_back(0);
      std::vector<double> &line_data = data.back();

      double value;
      std::istringstream iss(line);

      while (iss >> value)
      {
        line_data.push_back(value);
      }

      if (r > 1)
      {
        if ((int)line_data.size() < columns)
        {
          Output::root_warn("POD", "Row ", r, " has fewer than ",
            columns, " entries! Discarding.");

          data.pop_back();
        }
        else if ((int)line_data.size() > columns)
        {
          Output::root_warn("POD", "Row ", r, " has more than ",
            columns, " entries! Discarding.");

          data.pop_back();
        }
      }
      else
      {
        columns = (int)line_data.size();

        Output::info("POD", "Input file has ", columns, " columns.");
      }
    }

    ifile.close();
  }
}

/** ***************************************************************************/
void POD::write_pod()
{
  /* writing into basis into file */
  std::string pod_filename = basename + "pod";
  std::string name = "";

#ifdef _MPI
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  pod_filename += "_r" + std::to_string(mpi_rank);
#endif

  if (!filename_suffix.empty())
  {
    name += filename_suffix + "-";
  }

  if (db["pod_fluctuations_only"])
  {
    /* write averages of snapshots into file */
    write_averages();
  }

  Output::root_info<1>("POD", "Writing ", name, "POD basis into file: ", pod_filename);

  write_data(pod_filename, pod_basis.get_entries(), length_snaps, rank, db["pod_binary"]);

  /*write pod eigenvalues into file */
  write_eigenvalues();
}

/** ***************************************************************************/
void POD::read_basis(int rom_rank)
{
  this->rank = rom_rank;

  std::string filename = basename + "pod";

#ifdef _MPI
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  filename += "_r" + std::to_string(mpi_rank);
#endif

  if (this->db["pod_fluctuations_only"])
  {
    Output::root_info<1>("POD", "POD basis is computed out of fluctuating part of "
                     "snapshots!");

    read_averages();
  }

  std::vector<std::vector<double>> tmp_basis;
  read_data_ex(filename, tmp_basis, length_snaps, db["pod_binary"]);

  if (this->rank <= 0)
  {
    this->rank = (int)tmp_basis.size();
  }

  if (this->rank > (int)tmp_basis.size())
  {
    Output::root_warn<1>("POD basis", "Rank of POD basis can't be greater than ",
                    tmp_basis.size(), "! Parameter 'pod_rank' setting the "
                    "ROM dimension changed to ",
                    tmp_basis.size());

    this->rank = (int)tmp_basis.size();
  }

  this->length = (int)tmp_basis[0].size(); //total dof
  this->pod_basis = DenseMatrix(this->rank, this->length);

  std::string name;
  if (!filename_suffix.empty())
  {
    name += filename_suffix + "-";
  }

  Output::root_info<1>("POD", "  Length of ", name, "POD-basis vector : ", this->length);
  Output::root_info<1>("POD", "  Dimension of ", name, "POD basis     : ", this->rank);

  //pod basis (for all components)
  for (int i = 0; i < this->length; i++)
  {
    for (int j = 0; j < this->rank; j++)
    {
      this->pod_basis.setEntry(j,i, tmp_basis[j][i]);
    }
  }

  if (rank < (int)tmp_basis.size())
  {
    read_eigenvalues();
  }
}

/** ***************************************************************************/
void POD::read_eigenvalues()
{
  std::string eig_filename = basename + "eigs";

  Output::root_info<1>("POD", "Reading eigenvalues from file ", eig_filename);

  std::vector<double> eigenvalues;
  read_data(eig_filename, eigenvalues, 2);
  this->valid_eigs = eigenvalues.size();

  double sum_rest_eigs = 0.;
  for (int i = rank; i < valid_eigs; i++)
  {
    sum_rest_eigs += eigenvalues[i];
  }
  Output::root_info<1>("POD", "  Sum of the resting eigenvalues : ", sum_rest_eigs);
}

/** ***************************************************************************/
void POD::write_averages()
{
  std::string avr_filename = basename + "mean";

#ifdef _MPI
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  avr_filename += "_r" + std::to_string(mpi_rank);
#endif

  if ((int)this->snaps_mean.size() != this->length_snaps)
  {
    ErrThrow( "Error: Vector for averages of snapshots has the wrong length!" );
  }

  write_data(avr_filename, snaps_mean.data(), length_snaps, 1, db["pod_binary"]);
}

/** ***************************************************************************/
void POD::write_eigenvalues()
{
  std::string eig_filename = basename + "eigs";

  int valid_eigen = 0;
  int no_eigen = this->number_snaps;
  double sum_eigen = 0.0, cumulative=0.0;

  if (this->eigs == nullptr)
  {
    ErrThrow("Error: Eigenvalues not found. POD not created?");
  }

  for (int i = 0; i < no_eigen; i++)
  {
    if (this->eigs[i] >= this->eigen_threshold)
    {
      valid_eigen++;
    }
  }

  std::ofstream ofile;
  ofile.open(eig_filename.c_str(), std::ios::out | std::ios::trunc);
  ofile << setprecision(12);

  for (int i = 0; i < no_eigen; i++)
  {
    sum_eigen += this->eigs[i];
  }

  for (int i = 0; i < std::min(no_eigen,valid_eigen); i++)
  {
    cumulative += this->eigs[i] / sum_eigen;
    ofile << i + 1 << " \t" << this->eigs[i] << " \t" << cumulative << endl;
  }

  ofile.close();
}

/** ***************************************************************************/
void POD::read_averages()
{
  std::string avg_filename = basename + "mean";

#ifdef _MPI
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  avg_filename += "_r" + std::to_string(mpi_rank);
#endif

  Output::root_info<1>("POD", "Reading averages of snapshots from file ", avg_filename);

  std::vector<std::vector<double>> data;

  read_data_ex(avg_filename, data, length_snaps, db["pod_binary"]);

  if (data.size() > 0)
  {
    this->snaps_mean = data[0];

    if (data.size() > 1)
    {
      Output::root_warn("POD", "There appears to be more than one row of averages??");
    }
  }
  else
  {
    ErrThrow("No snapshot averages available!");
  }
}

/** ***************************************************************************/
void POD::read_data(std::string filename, std::vector<double>& data, int format)
{
  std::string line;
  std::ifstream file(filename.c_str());
  if (!file)
  {
    ErrThrow("Error: File ", filename, " could not be opened.");
  }

  while (getline(file, line))
  {
    double value;
    int column_i = 1;

    std::istringstream iss(line);
    while (iss >> value)
    {
      if (format == 0 || format == column_i)
      {
        data.push_back(value);
      }

      ++column_i;

      if (format != 0 && format < column_i)
      {
        break;
      }
    }
  }

  file.close();
}

/** ***************************************************************************/
void POD::compute_autocorr_mat(DenseMatrix& corr_mat)
{
  Output::root_info<1>("POD", "Computing autocorrelation matrix...");

  if (this->length_snaps == 0)
  {
    ErrThrow("Error: Snapshots are not available! Before computing POD "
             "basis read_snapshots() must be called!");
  }

  if (db["pod_inner_product"].is("euclidean"))
  {
  #ifdef _MPI
    Output::root_warn("POD", "POD with the Euclidean inner product will not "
      "give correct results in the MPI case since the POD class does not know "
      "what an FE space is. The inner product computed in this case is"
      "\n\n"
      "(phi_i, phi_j) = [number of ranks aware of both DOF #i and DOF #j]"
      "\n\n"
      "and so is dependent on the domain decomposition.");
  #endif

    bool transp_A = false;
    bool transp_B = true;

    corr_mat = *(snaps_mat.multiply( &this->snaps_mat, transp_A, transp_B ));
  }
  else
  {
    if (gramian_ptr == nullptr)
    {
      ErrThrow("POD with ", db["pod_inner_product"], " must be supplied with a "
        "Gramian matrix before calling compute_autocorr_mat.");
    }

    if ((int)this->gramian_ptr->get_n_rows() != this->length_snaps)
    {
      ErrThrow("ERROR: Dimension of inner product matrix and length of "
               "snapshots must coincide!\nCurrently: dim of matrix : ",
               this->gramian_ptr->get_n_rows(), " length of snapshots : ",
               this->length_snaps);
    }

    bool transpose = true;
    std::shared_ptr<DenseMatrix> tmp_mat = gramian_ptr->multiply(&snaps_mat,
                                                                 transpose);

    corr_mat = *(this->snaps_mat.multiply(tmp_mat.get()));
  } // end else

#ifdef _MPI
  MPI_Allreduce(MPI_IN_PLACE, corr_mat.get_entries(),
    this->number_snaps * this->number_snaps, MPI_DOUBLE,
    MPI_SUM, MPI_COMM_WORLD);
#endif

  // scale autocorrelation matrix with the number of snapshots according to
  //
  corr_mat.multiply(1. / this->number_snaps);
  Output::root_info<1>("POD", "... computation of autocorrelation matrix completed.");
}

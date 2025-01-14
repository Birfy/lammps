// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_bond_move.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "citeme.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix_bond_history.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "random_mars.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <stdio.h>

using namespace std;
using namespace LAMMPS_NS;
using namespace FixConst;

// static const char cite_fix_bond_swap[] =
//   "fix bond/swap command: doi:10.1063/1.1628670\n\n"
//   "@Article{Auhl03,\n"
//   " author = {R. Auhl and R. Everaers and G. S. Grest and K. Kremer and S. J. Plimpton},\n"
//   " title = {Equilibration of Long Chain Polymer Melts in Computer Simulations},\n"
//   " journal = {J.~Chem.\\ Phys.},\n"
//   " year =    2003,\n"
//   " volume =  119,\n"
//   " number =  12,\n"
//   " pages =   {12718--12728}\n"
//   "}\n\n";

#define DELTA_PERMUTE 100

/* ---------------------------------------------------------------------- */

FixBondMove::FixBondMove(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  tflag(0), alist(nullptr), id_temp(nullptr), type(nullptr), x(nullptr), list(nullptr),
  temperature(nullptr), random(nullptr)
{
  // if (lmp->citeme) lmp->citeme->add(cite_fix_bond_swap);

  if (narg != 8) error->all(FLERR,"Illegal fix bond/move command");

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/move command");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  fraction = utils::numeric(FLERR,arg[4],false,lmp);
  double cutoff = utils::numeric(FLERR,arg[5],false,lmp);
  cutsq = cutoff*cutoff;

  // initialize Marsaglia RNG with processor-unique seed

  int seed = utils::inumeric(FLERR,arg[6],false,lmp);
  random = new RanMars(lmp,seed + comm->me);

  tbondtype = utils::inumeric(FLERR,arg[7],false,lmp);

  // error check

  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix bond/move with non-molecular systems");

  // create a new compute temp style
  // id = fix-ID + temp, compute group = fix group

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} all temp",id_temp));
  tflag = 1;

  // initialize two permutation lists

  nmax = 0;
  alist = nullptr;

  maxpermute = 0;
  permute = nullptr;

  naccept = threesome = 0;
}

/* ---------------------------------------------------------------------- */

FixBondMove::~FixBondMove()
{
  delete random;

  // delete temperature if fix created it

  if (tflag) modify->delete_compute(id_temp);
  delete[] id_temp;

  memory->destroy(alist);
  delete[] permute;
}

/* ---------------------------------------------------------------------- */

int FixBondMove::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondMove::init()
{
  // require an atom style with molecule IDs

  if (atom->molecule == nullptr)
    error->all(FLERR,
               "Must use atom style with molecule IDs with fix bond/move");

  int icompute = modify->find_compute(id_temp);
  if (icompute < 0)
    error->all(FLERR,"Temperature ID for fix bond/move does not exist");
  temperature = modify->compute[icompute];

  // pair and bonds must be defined
  // no dihedral or improper potentials allowed
  // special bonds must be 0 1 1

  if (force->pair == nullptr || force->bond == nullptr)
    error->all(FLERR,"Fix bond/move requires pair and bond styles");

  if (force->pair->single_enable == 0)
    error->all(FLERR,"Pair style does not support fix bond/move");

  // if (force->angle == nullptr && atom->nangles > 0 && comm->me == 0)
  //   error->warning(FLERR,"Fix bond/move will not preserve correct angle "
  //                  "topology because no angle_style is defined");

  if (force->angle || force->dihedral || force->improper)
    error->all(FLERR,"Fix bond/move cannot use angle, dihedral or improper styles");

  if (force->special_lj[1] != 0.0 || force->special_lj[2] != 1.0 ||
      force->special_lj[3] != 1.0)
    error->all(FLERR,"Fix bond/move requires special_bonds = 0,1,1");

  // need a half neighbor list, built every Nevery steps

  neighbor->add_request(this, NeighConst::REQ_OCCASIONAL);

  // zero out stats

  naccept = threesome = 0;
  angleflag = 0;
  if (force->angle) angleflag = 1;
}

/* ---------------------------------------------------------------------- */

void FixBondMove::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ----------------------------------------------------------------------
   look for and perform swaps
   NOTE: used to do this every pre_neighbor(), but think that is a bug
         b/c was doing it after exchange() and before neighbor->build()
         which is when neigh lists are actually out-of-date or even bogus,
         now do it based on user-specified Nevery, and trigger reneigh
         if any swaps performed, like fix bond/create
------------------------------------------------------------------------- */

void FixBondMove::post_integrate()
{
  int i,j,ii,jj,kk,m,inum,jnum;
  int bondloc;
  int inext,iprev,ilast,jnext,jprev,jlast,ibond,iangle,jbond,jangle,inextbond;
  int ibondtype,jbondtype,iangletype,inextangletype,jangletype,jnextangletype;
  tagint itag,inexttag,iprevtag,ilasttag,jtag,jnexttag,jprevtag,jlasttag;
  int jjnext, iii;
  tagint i1,i2,i3,j1,j2,j3;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double delta,factor;

  if (update->ntimestep % nevery) return;

  // compute current temp for Boltzmann factor test

  double t_current = temperature->compute_scalar();

  // local ptrs to atom arrays

  tagint *tag = atom->tag;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int *num_angle = atom->num_angle;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;
  int **angle_type = atom->angle_type;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int newton_bond = force->newton_bond;
  int nlocal = atom->nlocal;

  type = atom->type;
  x = atom->x;

  neighbor->build_one(list,1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // randomize list of my owned atoms that are in fix group
  // grow atom list if necessary

  if (atom->nmax > nmax) {
    memory->destroy(alist);
    nmax = atom->nmax;
    memory->create(alist,nmax,"bondmove:alist");
  }

  // use randomized permutation of both I and J atoms in double loop below
  // this is to avoid any bias in accepted MC swaps based on
  //   ordering LAMMPS creates on a processor for atoms or their neighbors

  // create a random permutation of list of Neligible atoms
  // uses one-pass Fisher-Yates shuffle on an initial identity permutation
  // output: randomized alist[] vector, used in outer loop to select an I atom
  // similar randomized permutation is created for neighbors of each I atom

  int neligible = 0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (mask[i] & groupbit)
      alist[neligible++] = i;
  }

  int tmp;
  for (i = 0; i < neligible-1; i++) {
    j = i + static_cast<int> (random->uniform() * (neligible-i));
    tmp = alist[i];
    alist[i] = alist[j];
    alist[j] = tmp;
  }

  // examine ntest of my eligible atoms for potential swaps
  // atom I is randomly selected via atom list
  // look at all J neighbors of atom I
  //   done in a randomized permutation, via neighbor_permutation()
  // J must be on-processor (J < nlocal)
  // I,J must be in fix group
  // I,J must have same molecule IDs
  //   use case 1 (see doc page):
  //     if user defines mol IDs appropriately for linear chains,
  //     this will mean they are same distance from (either) chain end
  //   use case 2 (see doc page):
  //     if user defines a unique mol ID for desired bond sites (on any chain)
  //     and defines the fix group as these sites,
  //     this will mean they are eligible bond sites

  int ntest = static_cast<int> (fraction * neligible);
  int accept = 0;

  for (int itest = 0; itest < ntest; itest++) {
    i = alist[itest];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    neighbor_permutation(jnum);

    for (ibond = 0; ibond < num_bond[i]; ibond++) {
      inext = atom->map(bond_atom[i][ibond]);
      if (inext >= nlocal || inext < 0) continue;
      if ((mask[inext] & groupbit) == 0) continue;
      ibondtype = bond_type[i][ibond];

      if (ibondtype != tbondtype) continue;
      
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[permute[jj]];
        j &= NEIGHMASK;
        if (j >= nlocal) continue;
        if ((mask[j] & groupbit) == 0) continue;

        int findbond = 0;
        for (ibond = 0; ibond < num_bond[j]; ibond++) {
          if (bond_atom[j][ibond] == tag[inext])
            findbond = 1;
        }
        for (ibond = 0; ibond < num_bond[inext]; ibond++) {
          if (bond_atom[inext][ibond] == tag[j])
            findbond = 1;
        }

        if (findbond == 1) 
          goto first;
      }
      }
    }

  

 first:

  // trigger immediate reneighboring if swaps occurred on one or more procs

  int accept_any;
  MPI_Allreduce(&accept,&accept_any,1,MPI_INT,MPI_SUM,world);
  if (accept_any) next_reneighbor = update->ntimestep;

  if (!accept) return;
  naccept++;

  error->warning(FLERR,"Attemping to move the following bonds");
  error->warning(FLERR,std::to_string(i));
  error->warning(FLERR,std::to_string(num_bond[i]));
  error->warning(FLERR,std::to_string(bond_atom[i][0]));
  error->warning(FLERR,std::to_string(inext));
  error->warning(FLERR,std::to_string(num_bond[inext]));
  error->warning(FLERR,std::to_string(bond_atom[inext][0]));
  error->warning(FLERR,std::to_string(j));
  error->warning(FLERR,std::to_string(num_bond[j]));
  error->warning(FLERR,std::to_string(bond_atom[j][0]));

  // find instances of bond/history to reset history
  auto histories = modify->get_fix_by_style("BOND_HISTORY");
  int n_histories = histories.size();

  // change bond partners of affected atoms
  // on atom i: bond i-inext changes to i-jnext
  // on atom j: bond j-jnext changes to j-inext
  // on atom inext: bond inext-i changes to inext-j
  // on atom jnext: bond jnext-j changes to jnext-i

  if (bondloc == 0) {
    for (ibond = 0; ibond < num_bond[i]; ibond++)
    if (bond_atom[i][ibond] == tag[inext]) {
      if (n_histories > 0)
        for (auto &ihistory: histories)
          dynamic_cast<FixBondHistory *>(ihistory)->delete_history(i,ibond);
      bond_atom[i][ibond] = tag[j];
    }
  } else {
    bond_atom[i][num_bond[i]] = tag[j];
    bond_type[i][num_bond[i]] = tbondtype;
    num_bond[i]++;

    for (ibond = 0; ibond < num_bond[inext]; ibond++) {
      if (bond_atom[inext][ibond] == tag[i]) {
        if (n_histories > 0)
          for (auto &ihistory: histories)
            dynamic_cast<FixBondHistory *>(ihistory)->delete_history(inext,ibond);
        for (iii = ibond; iii < num_bond[inext] - 1; iii++) {
          bond_atom[inext][iii] = bond_atom[inext][iii+1];
          bond_type[inext][iii] = bond_type[inext][iii+1];
        }
        num_bond[inext]--;
      }
    }
  }

  itag = tag[i];
  inexttag = tag[inext];
  jtag = tag[j];

  // change 1st special neighbors of affected atoms: i,j,inext,jnext
  // don't need to change 2nd/3rd special neighbors for any atom
  //   since special bonds = 0 1 1 means they are never used

  for (m = 0; m < nspecial[i][0]; m++)
    if (special[i][m] == inexttag)
        special[i][m] = jtag;
    
  // for (m = 0; m < nspecial[j][0]; m++)
  //   if (special[j][m] == itag) {
    special[j][nspecial[j][0]] = itag;
    nspecial[j][0]++;
    // } 
  for (m = 0; m < nspecial[inext][0]; m++)
    if (special[inext][m] == itag) {
      for (iii = m; iii < nspecial[inext][0] - 1; iii++)
        special[inext][iii] = special[inext][iii+1];
      nspecial[inext][0]--;
    }
}
/* ---------------------------------------------------------------------- */

int FixBondMove::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    if (tflag) {
      modify->delete_compute(id_temp);
      tflag = 0;
    }
    delete[] id_temp;
    id_temp = utils::strdup(arg[1]);

    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find fix_modify temperature ID");
    temperature = modify->compute[icompute];

    if (temperature->tempflag == 0)
      error->all(FLERR,"Fix_modify temperature ID does not "
                 "compute temperature");
    if (temperature->igroup != igroup && comm->me == 0)
      error->warning(FLERR,"Group for fix_modify temp != fix group");
    return 2;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   compute squared distance between atoms I,J
   must use minimum_image since J was found thru atom->map()
------------------------------------------------------------------------- */

double FixBondMove::dist_rsq(int i, int j)
{
  double delx = x[i][0] - x[j][0];
  double dely = x[i][1] - x[j][1];
  double delz = x[i][2] - x[j][2];
  domain->minimum_image(delx,dely,delz);
  return (delx*delx + dely*dely + delz*delz);
}

/* ----------------------------------------------------------------------
   return pairwise interaction energy between atoms I,J
   will always be full non-bond interaction, so factors = 1 in single() call
------------------------------------------------------------------------- */

double FixBondMove::pair_eng(int i, int j)
{
  double tmp;
  double rsq = dist_rsq(i,j);
  return force->pair->single(i,j,type[i],type[j],rsq,1.0,1.0,tmp);
}

/* ---------------------------------------------------------------------- */

double FixBondMove::bond_eng(int btype, int i, int j)
{
  double tmp;
  double rsq = dist_rsq(i,j);
  return force->bond->single(btype,rsq,i,j,tmp);
}

/* ---------------------------------------------------------------------- */

double FixBondMove::angle_eng(int atype, int i, int j, int k)
{
  // test for non-existent angle at end of chain

  if (i == -1 || k == -1) return 0.0;
  return force->angle->single(atype,i,j,k);
}

/* ----------------------------------------------------------------------
   create a random permutation of one atom's N neighbor list atoms
   uses one-pass Fisher-Yates shuffle on an initial identity permutation
   output: randomized permute[] vector, used to index neighbors
------------------------------------------------------------------------- */

void FixBondMove::neighbor_permutation(int n)
{
  int i,j,tmp;

  if (n > maxpermute) {
    delete[] permute;
    maxpermute = n + DELTA_PERMUTE;
    permute = new int[maxpermute];
  }

  // Fisher-Yates shuffle

  for (i = 0; i < n; i++) permute[i] = i;

  for (i = 0; i < n-1; i++) {
    j = i + static_cast<int> (random->uniform() * (n-i));
    tmp = permute[i];
    permute[i] = permute[j];
    permute[j] = tmp;
  }
}

/* ----------------------------------------------------------------------
   return bond swapping stats
   n = 1 is # of swaps
   n = 2 is # of attempted swaps
------------------------------------------------------------------------- */

double FixBondMove::compute_vector(int n)
{
  double one,all;
  if (n == 0) one = naccept;
  else one = threesome;
  MPI_Allreduce(&one,&all,1,MPI_DOUBLE,MPI_SUM,world);
  return all;
}

/* ----------------------------------------------------------------------
   memory usage of alist
------------------------------------------------------------------------- */

double FixBondMove::memory_usage()
{
  double bytes = (double)nmax * sizeof(int);
  return bytes;
}

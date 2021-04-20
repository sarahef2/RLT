//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Estimate the Expectation of Variance
//  **********************************

// my header file
# include "RLT.h"
# include "Trees//Trees.h"
# include "Utility/Utility.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List EofVar_S(arma::umat& ObsTrack,
            arma::cube& Pred,
            arma::cube& Pred_TV,
            arma::uvec& C,
            int usecores,
            int verbose)
{
  DEBUG_Rcout << "-- calculate E(Var(Tree|C Shared)) ---" << std::endl;
  
  DEBUG_Rcout << C << std::endl;
  
  usecores = checkCores(usecores, verbose);
  
  size_t N = Pred.n_slices;
  size_t ntrees = Pred.n_cols;
  size_t length = C.n_elem;
  size_t tmpts = Pred.n_rows;
  
  arma::mat Tv_Est(N, tmpts, fill::zeros);
  arma::mat tmp(N, tmpts, fill::zeros);
  arma::cube Tree_Cov_Est(tmpts, tmpts, N, fill::zeros);
  for(size_t n = 0; n < N; n++){
    tmp = Pred_TV.slice(n);
    for(size_t m = 0; m < tmpts; m++){
      Tv_Est(n, m) = var(tmp.row(m));
    }
    Tree_Cov_Est.slice(n) = cov(tmp.t());
  }
  
  Rcout << "-- Initialize ---" << std::endl;
  arma::cube Est(length, tmpts, N, fill::zeros);
  arma::uvec allcounts(length, fill::zeros);
  arma::umat pairs(ntrees*ntrees, 4, fill::zeros);

  //Rcout << "Count pairs up to "<<ntrees*ntrees << std::endl;
  //size_t rnum = 0;
  //Rcout << "Start rnum: "<<rnum << std::endl;
//#pragma omp parallel num_threads(usecores)
//{
//#pragma omp for schedule(dynamic)
  for (size_t i = 0; i < (ntrees - 1); i++){
    for (size_t j = i+1; j < ntrees; j++){
      uvec pair = {i, j};
      int  incommon = sum( min(ObsTrack.cols(pair), 1) );
      //uvec tmppair = {i, j, incommon};
      //Rcout << "Incommon count: "<< incommon << std::endl;
      //Rcout << "Row num: "<<i*(ntrees-1)+(j-1-i) << std::endl;
      pairs(i*(ntrees-1)+(j-1-i), 0) = i;
      pairs(i*(ntrees-1)+(j-1-i), 1) = j;
      pairs(i*(ntrees-1)+(j-1-i), 2) = incommon-C(0);
      //pairs(i*(ntrees-1)+(j-1-i), 3) = incommon-C(0);
      if ((incommon-C(0))<length){
        //Rcout << "Index: "<<incommon - C(0) << std::endl;
        pairs(i*(ntrees-1)+(j-1-i), 3) = 1;
        allcounts(incommon - C(0))++;
      }
      //rnum++;
      //Rcout << rnum << std::endl;
      //if ( C(0)<=incommon and C(length-1)>=incommon){
      //    allcounts(incommon - length)++;
      //    pairs.insert_rows(0, tmppair);
      //  }
      }
    }
//}
  
  //arma::uvec wi_C = find(pairs.col(3)>=C(0) and pairs.col(3)<=C(length-1));
  //arma::umat pairs_red = pairs.rows(wi_C);
  
  size_t pair_count = pairs.n_rows;
  //Rcout << "Pair counts "<< pair_count << std::endl;
  //Rcout << "pairs "<< pairs.rows(0,4) << std::endl;
  //arma::mat wcsigma(N, tmpts);
  arma::cube wcsigmaCov(tmpts, tmpts, N);
  arma::mat temp2;
  
  //Rcout << "Starting paralell computing" << std::endl;
//#pragma omp parallel num_threads(usecores)
  //{
  //#pragma omp for schedule(dynamic)
    for(size_t n = 0; n < N; n++){
      //Rcout << "n: "<<n << std::endl;
      arma::cube Cov_Est(tmpts, tmpts, length, fill::zeros);
      for(size_t k = 0; k < pair_count; k++){
        //Rcout << "k: "<<k << std::endl;
        //Rcout << "pairs(k, 2): "<<pairs(k, 2) << " pairs(k, 3): "<<pairs(k, 3) << std::endl;
        if(pairs(k, 3)==1){
          //Rcout << "k: "<<k << std::endl;
          for(size_t tm = 0; tm < tmpts; tm++){
             Est(pairs(k, 2), tm, n) += 0.5 * (Pred(tm,pairs(k, 0),n) - Pred(tm,pairs(k, 1),n)) *
               (Pred(tm,pairs(k, 0),n) - Pred(tm,pairs(k, 1),n))/allcounts(pairs(k,2));
            for(size_t tm2 = 0; tm2 < tmpts; tm2++){
              Cov_Est(tm, tm2, pairs(k, 2)) += 0.5 * (Pred(tm,pairs(k, 0),n) - Pred(tm2,pairs(k, 1),n)) *
                (Pred(tm,pairs(k, 0),n) - Pred(tm2,pairs(k, 1),n))/allcounts(pairs(k,2));
              }
            }
        }
      }
      //Rcout << "Cov_Est "<< Cov_Est.slice(0) << std::endl;
      arma::mat test = sum(Cov_Est, 2);
      //Rcout << "Sum over slices "<< test << std::endl;
      //Rcout << "Dim of Sum over slices "<< test.n_rows <<" "<<test.n_cols << std::endl;
      //Rcout << "Dim of slice "<< wcsigmaCov.slice(n).n_rows <<" "<<wcsigmaCov.slice(n).n_cols << std::endl;
      wcsigmaCov.slice(n) = sum(Cov_Est, 2);
    }
  //}
  
  // #pragma omp parallel num_threads(usecores)
  // {
  // #pragma omp for schedule(dynamic)
  //       for (size_t l = 0; l < length; l++){ // calculate all C values
  //         arma::cube Cov_Est(tmpts, tmpts, N, fill::zeros);
  //         size_t count = 0;
  //     
  //     for (size_t i = 0; i < (ntrees - 1); i++){
  //       for (size_t j = i+1; j < ntrees; j++){
  //         
  //         uvec pair = {i, j};
  //         pairs.insert_rows(0, {i, j, C(l)});
  //         
  //         if ( sum( min(ObsTrack.cols(pair), 1) ) == C(l) )
  //         {
  //           count++;
  //           for(size_t n = 0; n < N; n++){
  //             for(size_t tm = 0; tm < tmpts; tm++){
  //                 Est(l, tm, n) += 0.5 * (Pred(tm,i,n) - Pred(tm,j,n)) *
  //                 (Pred(tm,i,n) - Pred(tm,j,n));
  //               for(size_t tm2 = 0; tm2 < tmpts; tm2++){
  //                 Cov_Est(tm, tm2, n) += 0.5 * (Pred(tm,i,n) - Pred(tm2,j,n)) *
  //                   (Pred(tm,i,n) - Pred(tm2,j,n));
  //               }
  //             }
  //           }
  //         }
  //       }}
  //     
  //     Est.row(l) /= count;
  //     allcounts(l) = count;
  //     Cov_Est /= count;
  //   }
  // }

  DEBUG_Rcout << "-- total count  ---" << allcounts << std::endl;  
  DEBUG_Rcout << "-- all estimates  ---" << Est << std::endl; 
  
  // For a given test ob
  // Est.slice(n) length*tmpt
  // allcounts length
  // Want vector of length tmpt
  // arma::mat wcsigma(N, tmpts);
  // arma::mat temp2;

  // for(size_t n = 0; n < N; n++){
  //   temp2 = Est.slice(n);
  //   for(size_t tm = 0; tm < tmpts; tm++){
  //     wcsigma(n,tm) = sum(temp2.col(tm));// * temp2.col(tm));
  //   }
  // }
  
  //Why do we want to multiply by all counts?
  
  //Rcout << "Matrix manipulation" << std::endl;
  arma::mat wcsigma = sum(Est, 0);
  
  arma::mat var = Tv_Est - wcsigma/sum(allcounts);
  arma::cube cov = Tree_Cov_Est - wcsigmaCov/sum(allcounts);

  List ReturnList;
  
  ReturnList["allcounts"] = allcounts;
  ReturnList["estimation"] = Est;
  ReturnList["tree.var"] = Tv_Est;
  ReturnList["wcsigma"] = wcsigma;
  ReturnList["wcsigmaCov"] = wcsigmaCov;
  ReturnList["var"] = var;
  ReturnList["tree.cov"] = Tree_Cov_Est;
  ReturnList["cov"] = cov;
  
  return(ReturnList);
}






// List EofVar_Surv(//arma::cube& Pred,
//             //arma::cube& Pred_TV,
//             arma::umat& ObsTrack,
//             arma::mat& Pred,
//             arma::uvec& C,
//             int usecores,
//             int verbose)
// {
//   DEBUG_Rcout << "-- calculate E(Var(Tree|C Shared)) ---" << std::endl;
//   
//   DEBUG_Rcout << C << std::endl;
//   
//   usecores = checkCores(usecores, verbose);
//   
//   size_t N = Pred.n_rows;
//   size_t ntrees = Pred.n_cols;
//   //The number of C's with which we will estimate the variance
//   size_t length = C.n_elem;
// 
//   //arma::mat Tv_Est(N, Pred_TV.n_rows, fill::zeros);
//   //arma::mat tmp(N, Pred_TV.n_rows, fill::zeros);
//   //for(size_t n = 0; n < N; n++){
//   //  tmp = Pred_TV.slice(n);
//   //  for(size_t m = 0; m < Pred_TV.n_rows; m++){
//   //    Tv_Est(n, m) = var(tmp.row(m));
//   //  }
//   //}
//   
//   //For each observation, record the variance at each C  
//   arma::mat Est(N, length, fill::zeros);
//   //Keep track of the number of tree pairs used to calculate each C
//   arma::uvec allcounts(length, fill::zeros);
//    
// #pragma omp parallel num_threads(usecores)
// {
//   #pragma omp for schedule(dynamic)
//   for (size_t l = 0; l < length; l++) // calculate all C values
//   {
//     size_t count = 0;
//     
//     //For each pair of trees...
//     for (size_t i = 0; i < (ntrees - 1); i++){
//     for (size_t j = i+1; j < ntrees; j++){
//       
//       //Indices of the pair
//       uvec pair = {i, j};
//         
//         //Pulls the columns related to the indices
//         //Finds the minimum in each row
//         //If the minimum is 1, then that observation was included in both rows
//         //Count the number of obs used in both trees
//         //If the sum of shared obs equals C(l)...
//       if ( sum( min(ObsTrack.cols(pair), 1) ) == C(l) )
//       {
//         count++;
//         
//         //Calculate ..sigma_c and add it to the others
//         Est.col(l) += 0.5 * square(Pred.col(i) - Pred.col(j));
//       }
//     }}
//     
//     //Take the mean of ..sigma_c
//     Est.col(l) /= count;
//     //Keep the count of ..sigma_c's
//     allcounts(l) = count;
//   }
//   //We have now estimated \binom{n}{k}^{-2}sum(sum(..sigma_c))
// }
// 
//   DEBUG_Rcout << "-- total count  ---" << allcounts << std::endl;  
//   DEBUG_Rcout << "-- all estimates  ---" << Est << std::endl; 
// 
//   List ReturnList;
//   
//   ReturnList["allcounts"] = allcounts;
//   ReturnList["estimation"] = Est;
//   //ReturnList["tree.var"] = Tv_Est;
//   
//   return(ReturnList);

//}









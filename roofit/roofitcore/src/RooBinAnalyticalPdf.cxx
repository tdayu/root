// Authors: Stephan Hageboeck, CERN; Andrea Sciandra, SCIPP-UCSC/Atlas; Nov 2020

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2020, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
 * \class RooBinAnalyticalPdf
 * The RooBinAnalyticalPdf is supposed to be used as an adapter between a continuous PDF
 * and a binned distribution.
 * When RooFit is used to fit binned data, and the PDF is continuous, it takes the probability density
 * at the bin centre as a proxy for the probability averaged (integrated) over the entire bin. This is
 * correct only if the second derivative of the function vanishes, though.
 *
 * For PDFs that have larger curvatures, the RooBinAnalyticalPdf can be used. It integrates the PDF in each
 * bin using an adaptive integrator. This usually requires 21 times more function evaluations, but significantly
 * reduces biases due to better sampling of the PDF. The integrator can be accessed from the outside
 * using integrator(). This can be used to change the integration rules, so less/more function evaluations are
 * performed. The target precision of the integrator can be set in the constructor.
 *
 * \note This feature is currently limited to one-dimensional PDFs.
 *
 * ### How to use it
 * There are two ways to use this class:
 * - Manually wrap a PDF:
 * ```
 *   RooBinAnalyticalPdf binSampler("<name>", "title", <binned observable of PDF>, <original PDF> [, <precision for integrator>]);
 *   binSampler.fitTo(data);
 * ```
 *   When a PDF is wrapped with a RooBinAnalyticalPdf, just use the bin sampling PDF instead of the original one for fits
 *   or plotting etc. Note that the binning will be taken from the observable.
 * - Instruct test statistics to carry out this wrapping automatically:
 * ```
 *   pdf.fitTo(data, IntegrateBins(<precision>));
 * ```
 *   This method is especially useful when used with a simultaneous PDF, since each component will automatically be wrapped,
 *   depending on the value of `precision`:
 *   - `precision < 0.`: None of the PDFs are touched, bin sampling is off.
 *   - `precision = 0.`: Continuous PDFs that are fit to a RooDataHist are wrapped into a RooBinAnalyticalPdf. The target precision
 *      forwarded to the integrator is 1.E-4 (the default argument of the constructor).
 *   - `precision > 0.`: All continuous PDFs are automatically wrapped into a RooBinAnalyticalPdf. The `'precision'` is used for all
 *      integrators.
 *
 * ### Simulating a binned fit using RooDataSet
 *   Some frameworks use unbinned data (RooDataSet) to simulate binned datasets. By adding one entry for each bin centre with the
 *   appropriate weight, one can achieve the same result as fitting with RooDataHist. In this case, however, RooFit cannot
 *   auto-detect that a binned fit is running, and that an integration over the bin is desired (note that there are no bins to
 *   integrate over in this kind of dataset).
 *
 *   In this case, `IntegrateBins(>0.)` needs to be used, and the desired binning needs to be assigned to the observable
 *   of the dataset:
 *   ```
 *     RooRealVar x("x", "x", 0., 5.);
 *     x.setBins(10);
 *
 *     // <create dataset and model>
 *
 *     model.fitTo(data, IntegrateBins(>0.));
 *   ```
 *
 *   \see RooAbsPdf::fitTo()
 *   \see IntegrateBins()
 *
 *
 * \htmlonly <style>div.image img[src="RooBinAnalyticalPdf_OFF.png"]{width:15cm;}</style> \endhtmlonly
 * \htmlonly <style>div.image img[src="RooBinAnalyticalPdf_ON.png" ]{width:15cm;}</style> \endhtmlonly
 * <table>
 * <tr><td>
 * \image html RooBinAnalyticalPdf_OFF.png "Binned fit without RooBinAnalyticalPdf"
 * <td>
 * \image html RooBinAnalyticalPdf_ON.png "Binned fit with RooBinAnalyticalPdf"
 * </table>
 *
 */


#include "RooBinAnalyticalPdf.h"

#include "RooHelpers.h"
#include "RooRealBinding.h"
#include "BatchHelpers.h"
#include "RunContext.h"
#include "RooRangeBinning.h"

#include "Math/Integrator.h"

#include <algorithm>

////////////////////////////////////////////////////////////////////////////////
/// Construct a new RooBinAnalyticalPdf.
/// \param[in] name A name to identify this object.
/// \param[in] title Title (for e.g. plotting)
/// \param[in] observable Observable to integrate over (the one that is binned).
/// \param[in] inputPdf A PDF whose bins should be sampled with higher precision.
/// \param[in] epsilon Relative precision for the integrator, which is used to sample the bins.
/// Note that ROOT's default is to use an adaptive integrator, which in its first iteration usually reaches
/// relative precision of 1.E-4 or better. Therefore, asking for lower precision rarely has an effect.
RooBinAnalyticalPdf::RooBinAnalyticalPdf(const char *name, const char *title, RooRealVar& observable,
    RooAbsPdf& inputPdf, const RooArgSet * normSet, const char * rangeName) :
      RooAbsPdf(name, title),
      _pdf("inputPdf", "Function to be converted into a PDF", this, inputPdf),
      _observable("observable", "Observable to integrate over", this, observable, true, true),
      _integral("integral", "Analytical integral object", this, true, false, true) {
  if (!_pdf->dependsOn(*_observable)) {
    throw std::invalid_argument(std::string("RooBinAnalyticalPdf(") + GetName()
        + "): The PDF " + _pdf->GetName() + " needs to depend on the observable "
        + _observable->GetName());
  }
  createAnalyticalIntegral(normSet, rangeName);
}

 ////////////////////////////////////////////////////////////////////////////////
 /// Copy a RooBinAnalyticalPdf.
 /// \param[in] other PDF to copy.
 /// \param[in] name Optionally rename the copy.
 RooBinAnalyticalPdf::RooBinAnalyticalPdf(const RooBinAnalyticalPdf& other, const char* name) :
   RooAbsPdf(other, name),
   _pdf("inputPdf", this, other._pdf),
   _observable("observable", this, other._observable),
   _integral("integral", this, other._integral) { }

void RooBinAnalyticalPdf::createAnalyticalIntegral(const RooArgSet * normSet, const char * rangeName) {
  auto binRange = RooRangeBinning(_binSamplingRangeName);
  _observable->setBinning(binRange, _binSamplingRangeName);
  auto * integral = _pdf->createIntegral(RooArgSet(*_observable), *normSet, rangeName);
  _integral.setArg(*integral);
}

////////////////////////////////////////////////////////////////////////////////
/// Integrate the PDF over the current bin of the observable.
double RooBinAnalyticalPdf::evaluate() const {
  const unsigned int bin = _observable->getBin();
  const double low = _observable->getBinning().binLow(bin);
  const double high = _observable->getBinning().binHigh(bin);

  // Important: When the integrator samples x, caching of sub-tree values needs to be off.
  RooHelpers::DisableCachingRAII disableCaching(inhibitDirty());

  auto& samplingBin = _observable->getBinning(_binSamplingRangeName);
  samplingBin.setRange(low, high);
  return _integral->getVal();
}

////////////////////////////////////////////////////////////////////////////////
/// Integrate the PDF over all its bins, and return a batch with those values.
/// \param[in/out] evalData Struct with evaluation data.
/// \param[in] normSet Normalisation set that's used to evaluate the PDF.
RooSpan<double> RooBinAnalyticalPdf::evaluateSpan(BatchHelpers::RunContext& evalData, const RooArgSet* normSet) const {
  // Retrieve binning, which we need to compute the probabilities
  auto boundaries = binBoundaries();
  auto xValues = _observable->getValues(evalData, normSet);
  auto results = evalData.makeBatch(this, xValues.size());
  auto& samplingBin = _observable->getBinning(_binSamplingRangeName);

  // Important: When the integrator samples x, caching of sub-tree values needs to be off.
  RooHelpers::DisableCachingRAII disableCaching(inhibitDirty());

  // Now integrate PDF in each bin:
  for (unsigned int i=0; i < xValues.size(); ++i) {
    const double x = xValues[i];
    const auto upperIt = std::upper_bound(boundaries.begin(), boundaries.end(), x);
    const unsigned int bin = std::distance(boundaries.begin(), upperIt) - 1;
    assert(bin < boundaries.size());

    samplingBin.setRange(boundaries[bin], boundaries[bin+1]);
    results[i] = _integral->getVal(normSet);
  }

  return results;
}

////////////////////////////////////////////////////////////////////////////////
/// Get the bin boundaries for the observable.
/// These will be recomputed whenever the shape of this object is dirty.
RooSpan<const double> RooBinAnalyticalPdf::binBoundaries() const {
  if (isShapeDirty() || _binBoundaries.empty()) {
    _binBoundaries.clear();
    const RooAbsBinning& binning = _observable->getBinning(nullptr);
    const double* boundaries = binning.array();

    for (int i=0; i < binning.numBoundaries(); ++i) {
      _binBoundaries.push_back(boundaries[i]);
    }

    assert(std::is_sorted(_binBoundaries.begin(), _binBoundaries.end()));

    clearShapeDirty();
  }

  return {_binBoundaries};
}

////////////////////////////////////////////////////////////////////////////////
/// Return a list of all bin boundaries, so the PDF is plotted correctly.
/// \param[in] obs Observable to generate the boundaries for.
/// \param[in] xlo Beginning of range to create list of boundaries for.
/// \param[in] xhi End of range to create to create list of boundaries for.
/// \return Pointer to a list to be deleted by caller.
std::list<double>* RooBinAnalyticalPdf::binBoundaries(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const {
  if (obs.namePtr() != _observable->namePtr()) {
    coutE(Plotting) << "RooBinAnalyticalPdf::binBoundaries(" << GetName() << "): observable '" << obs.GetName()
        << "' is not the observable of this PDF ('" << _observable->GetName() << "')." << std::endl;
    return nullptr;
  }

  auto list = new std::list<double>;
  for (double val : binBoundaries()) {
    if (xlo <= val && val < xhi)
      list->push_back(val);
  }

  return list;
}

////////////////////////////////////////////////////////////////////////////////
/// Return a list of all bin centres, so the PDF is plotted correctly.
/// \param[in] obs Observable to generate the sampling hint for.
/// \param[in] xlo Beginning of range to create sampling hint for.
/// \param[in] xhi End of range to create sampling hint for.
/// \return Pointer to a list to be deleted by caller.
std::list<double>* RooBinAnalyticalPdf::plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const {
  if (obs.namePtr() != _observable->namePtr()) {
    coutE(Plotting) << "RooBinAnalyticalPdf::plotSamplingHint(" << GetName() << "): observable '" << obs.GetName()
        << "' is not the observable of this PDF ('" << _observable->GetName() << "')." << std::endl;
    return nullptr;
  }

  auto binCentres = new std::list<double>;
  const auto& binning = obs.getBinning();

  for (unsigned int bin=0; bin < static_cast<unsigned int>(binning.numBins()); ++bin) {
    const double centre = binning.binCenter(bin);

    if (xlo <= centre && centre < xhi)
      binCentres->push_back(centre);
  }

  return binCentres;
}

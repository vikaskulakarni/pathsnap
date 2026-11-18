"""Build the technical report from canonical result tables and optional rehearsal images."""
import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WIDTH, HEIGHT = A4


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Report:
    def __init__(self, path):
        self.pdf = canvas.Canvas(str(path), pagesize=A4)
        self.pdf.setTitle("PathSnap | Route reconstruction")
        self.y = 0

    def page(self, section, title):
        if self.y:
            self.pdf.showPage()
        self.pdf.setFillColor(colors.HexColor("#332947"))
        self.pdf.rect(0, HEIGHT-90, WIDTH, 90, fill=1, stroke=0)
        self.pdf.setFillColor(colors.white)
        self.pdf.setFont("Helvetica", 10)
        self.pdf.drawString(38, HEIGHT-30, "PATHSNAP   /   " + section.upper())
        self.pdf.setFont("Helvetica-Bold", 24)
        self.pdf.drawString(38, HEIGHT-65, title)
        self.pdf.setFont("Helvetica", 9)
        self.pdf.setFillColor(colors.HexColor("#695c74"))
        self.pdf.drawRightString(WIDTH-38, 24, f"PathSnap   /   {self.pdf.getPageNumber()}")
        self.y = HEIGHT-109

    def para(self, text, bold=False, size=10):
        style = ParagraphStyle("body", fontName="Helvetica-Bold" if bold else "Helvetica",
                               fontSize=size, leading=size*1.4, textColor=colors.HexColor("#27232b"))
        item = Paragraph(escape(text), style)
        _, height = item.wrap(WIDTH-76, 720)
        self.y -= height
        if self.y < 32:
            raise ValueError(f"Text overflows page {self.pdf.getPageNumber()}")
        item.drawOn(self.pdf, 38, self.y)
        self.y -= 8

    def table(self, headers, rows, widths):
        for index, row in enumerate([headers, *rows]):
            self.pdf.setFillColor(colors.HexColor("#ede8f2" if index%2==0 else "#faf8fc"))
            self.pdf.rect(38, self.y-22, sum(widths), 22, fill=1, stroke=0)
            self.pdf.setFillColor(colors.HexColor("#292334"))
            self.pdf.setFont("Helvetica-Bold" if index==0 else "Helvetica", 9)
            left=44
            for cell, width in zip(row,widths):
                self.pdf.drawString(left,self.y-14,str(cell))
                left+=width
            self.y-=22
        self.y-=12

    def model_scores(self, models, names):
        """Draw paired bars directly from student-test metrics in the experiment CSV."""
        left, span = 152, 306
        self.para("Student-test Top-1 accuracy and candidate macro F1", True, 11)
        for label, color, x in [("Top-1", "#3b6ea5", left), ("Macro F1", "#b05b29", left+78)]:
            self.pdf.setFillColor(colors.HexColor(color))
            self.pdf.rect(x,self.y-7,8,8,stroke=0,fill=1)
            self.pdf.setFont("Helvetica",8)
            self.pdf.drawString(x+12,self.y-6,label)
        self.y -= 16
        for row in models:
            self.pdf.setFont("Helvetica",8)
            self.pdf.setFillColor(colors.black)
            self.pdf.drawString(39,self.y-7,names[row["model"]])
            for metric,color in [("test_top1","#3b6ea5"),("test_macro_f1","#b05b29")]:
                value=float(row[metric])
                self.pdf.setFillColor(colors.HexColor(color))
                self.pdf.rect(left,self.y-6,span*value,5,stroke=0,fill=1)
                self.pdf.drawString(left+span*value+3,self.y-6,f"{value:.3f}")
                self.y-=8
            self.y-=6
        self.para("Bars start at zero; a full-width bar represents 1.0.", size=8)

    def image(self, path, caption, max_h=198):
        image = ImageReader(str(path))
        w,h=image.getSize()
        factor=min((WIDTH-76)/w,max_h/h)
        self.y-=h*factor
        if self.y<46:
            raise ValueError(f"Image overflows page {self.pdf.getPageNumber()}")
        self.pdf.drawImage(image,(WIDTH-w*factor)/2,self.y,w*factor,h*factor)
        self.y-=5
        self.para(caption,size=9)


def build(output, evidence):
    output.parent.mkdir(parents=True,exist_ok=True)
    doc=Report(output)
    models=records(RESULTS/"model_comparison.csv")
    tuning=records(RESULTS/"tuning_comparison.csv")
    hmm=records(RESULTS/"hmm_results.csv")
    meta=json.loads((ROOT/"artifacts/best_params.json").read_text())
    figs=RESULTS/"figures"
    pct=lambda v: f"{float(v)*100:.2f}%"
    names={"hist_gradient_boosting":"Histogram boosting","random_forest":"Random forest",
           "decision_tree":"Decision tree","logistic_regression":"Logistic regression",
           "gaussian_naive_bayes":"Gaussian naive Bayes"}
    doc.page("Development evaluation","Route reconstruction from GPS")
    doc.para("How much does road connectivity help when GPS alone leaves the direction of travel ambiguous?",True,16)
    doc.para("The system ranks nearby directed road segments for each observation, then uses Viterbi decoding to select a path. The browser on Laptop 2 submits a trip ID. Laptop 1 builds features, scores candidates, decodes the route, and evaluates the completed prediction against its local reference.")
    doc.table(["Student-test method","Top-1","Continuity"],[[r["method"],pct(r["top1"]),pct(r["route_continuity"])] for r in hmm if r["method"] in ("nearest_arc","classifier_only(model)","hmm_classifier(model)")],[245,122,148])
    doc.para("Across 10 student-test trips and 19,160 observations, the deployed decoder improved pointwise accuracy by 8.30 percentage points over its underlying classifier. Both methods faced the same GPS observations and road networks.")
    doc.para("Data boundaries",True,14)
    doc.para("The inventory covers 80 training trips and 10 student-test trips. Training labels come from aligning GPS points to an ordered reference route. Seven training trips exceeded the team's label-quality cutoff (95th percentile distance above 30 m); the deployed model comparison used the other 73. The earlier starter baseline used all 80 trips. Its score belongs to a separate run.")
    doc.para("The 10 student-test trips support development comparisons. Reserved evaluator trips were excluded from this rehearsal. Results throughout come from the active results/ directory and the deployed model identified by artifacts/best_params.json.")
    doc.page("Exploration","What the observations reveal")
    doc.para("All 90 development trips had a track, nodes, arcs and reference route. The scan found no missing numeric values, invalid coordinates or non-increasing timestamps. Road graphs differed by trip; 79 training graphs and all 10 student-test graphs contained multiple weak components.")
    doc.image(figs/"eda_time_gaps_train.png","Training intervals cluster near one second; 1,216 exceed five seconds. The longest gap spans about 42.5 hours.",160)
    doc.image(figs/"eda_gps_noise_train.png","Training distance to the nearest mapped segment: median 4.97 m and 95th percentile 83.43 m. On student-test these were 4.76 m and 16.00 m.",160)
    doc.para("A nearby mapped road may carry traffic in the opposite direction. The distance distribution motivates a fallback when local candidates are absent. Across long gaps, the decoder permits longer graph transitions.")
    doc.page("Preprocessing","From a trip to candidate rows")
    doc.para("The track records longitude, latitude and time. Nodes locate intersections and arc endpoints. Directed arcs link node IDs. A reference route lists arcs in travel order. The reference alignment minimises total point-to-segment distance while keeping progress along that ordered route non-decreasing.")
    doc.para("Coordinates are projected into metres. The pipeline retains up to eight nearest candidates inside 60 m before adding reverse-direction counterparts. A nearest-road fallback handles empty searches. Each candidate gets a binary label after feature construction. On student-test there were 122,193 candidate rows and 71 observations whose reference arc was missing.")
    doc.para("Twelve classifier inputs",True,14)
    doc.para("Distance, distance rank and distance above the nearest candidate measure proximity. Heading discrepancy and speed distinguish opposite directions. Time gap, arc length, position along the segment, fallback status, low-speed status, presence of a reverse arc and connection to the previous best candidate provide context. Trip IDs, raw positions and labels stay outside the learned feature vector.")
    doc.image(figs/"eda_network_train.png","Training graph structure. Node degree counts incident arcs; weak components count disconnected pieces when direction is ignored.",145)
    doc.image(figs/"eda_trip_stats_train.png","Training trip duration, observed track length and positive interval speeds. Entire trips remain together in validation.",130)
    doc.page("Model experiments","Five classifiers on the same trips")
    doc.para("Five-fold GroupKFold separates training trips. Top-1 selects one candidate per observation. Macro F1 and log loss use candidate rows. Continuity measures successive road choices. Timing below is milliseconds per 1,000 candidate rows.")
    doc.table(["Family / model","CV F1","Top-1","Test F1","Continuity","Loss","ms/1k"],[[names[r["model"]],f'{float(r["cv_macro_f1"]):.3f}',pct(r["test_top1"]),f'{float(r["test_macro_f1"]):.3f}',pct(r["test_route_continuity"]),f'{float(r["test_log_loss"]):.3f}',f'{float(r["infer_ms_per_1k_rows"]):.2f}'] for r in models],[134,49,66,58,76,60,72])
    notes={"hist_gradient_boosting":"Boosted shallow trees minimise classification log loss and capture feature interactions. Shrinkage and leaf size control complexity. It led CV Top-1 and was selected for tuning.",
           "random_forest":"Averaging bootstrapped trees reduces variance. Trees use impurity reduction. Depth and minimum leaf size limit memorisation, at the cost of smoothing rare patterns.",
           "decision_tree":"A single tree splits on impurity reduction and can express simple distance/heading thresholds. It is fast at inference but can be sensitive to the particular trips used for fitting.",
           "logistic_regression":"A scaled linear classifier minimises regularised log loss. Its coefficients combine clues efficiently; a linear boundary can miss interactions near junctions.",
           "gaussian_naive_bayes":"Gaussian class-conditional likelihoods make fitting fast. Dependence between distance, rank and relative distance challenges its independence assumption; its observed log loss is highest."}
    for row in models:
        doc.para(names[row["model"]]+"  |  "+row["configuration"],True,10)
        doc.para(notes[row["model"]]+" Grouped CV Top-1 "+pct(row["cv_top1"])+"; student-test "+pct(row["test_top1"])+".",size=9.5)
    doc.para("All five scored lower on student-test Top-1 than in grouped CV. Different trip populations can contribute to this gap; these two scores alone do not establish overfitting.",size=9)
    doc.page("Model diagnostics","Discrimination and error cost")
    for fname,cap in [("model_cv_macro_f1","Grouped training-trip CV macro F1 across families."),("model_student_log_loss","Candidate-level probability loss penalises confident errors.")]:
        doc.image(figs/(fname+".png"),cap,155)
    doc.model_scores(models,names)
    doc.page("Bayesian optimisation","Tuning the deployed classifier")
    search=meta["search"]
    doc.para(f'Optuna TPE ran {search["n_trials"]} trials with seed {search["seed"]}. To bound cost, the search used {search["folds"]} grouped folds on {search["trips"]} training trips. The objective was mean observation-level Top-1. The best search score was {pct(search["best_objective"])} in {search["seconds"]:.1f} seconds. The full training comparison below uses its own folds.')
    doc.table(["Parameter","Search range","Selected"],[[key,ran,f'{meta["best_params"][key]:.5g}'] for key,ran in [("learning_rate","0.02 to 0.3, log"),("max_iter","60 to 250"),("max_leaf_nodes","15 to 63"),("min_samples_leaf","20 to 200"),("l2_regularization","0.0001 to 10, log")]],[190,170,155])
    doc.table(["Configuration","CV Top-1","Test Top-1","Test F1","Continuity","Log loss"],[[r["configuration"],pct(r["cv_top1"]),pct(r["test_top1"]),r["test_macro_f1"],pct(r["test_route_continuity"]),r["test_log_loss"]] for r in tuning],[124,78,78,68,82,85])
    doc.para("Tuning raised grouped CV Top-1 slightly. Student-test Top-1 moved from 85.637% to 85.600%; continuity and macro F1 rose slightly and log loss worsened. The tuned artifact was used for the decoder comparison.")
    doc.table(["Configuration","CV macro F1","Inference ms / 1k rows"],[[r["configuration"],r["cv_macro_f1"],r["infer_ms_per_1k_rows"]] for r in tuning],[170,145,200])
    doc.para("The search varies learning rate and iterations together to control boosting strength, leaf count and minimum leaf size to control local complexity, and L2 regularisation to shrink leaf estimates.",size=9)
    doc.image(figs/"optuna_history.png","Trial objective over the Bayesian search. Each point is a grouped validation measurement.",150)
    doc.page("Sequence decoding","From independent choices to a route")
    doc.para("A hidden state is a candidate directed arc at one observation. The classifier emission blends log positive-class probability with a Gaussian distance score (weight 0.5, distance scale 8 m). Transitions favour staying on the same arc, then a directed successor; longer hops and U-turns receive lower scores.")
    doc.para("Viterbi retains the highest cumulative score for each current arc and backtracks from the best final state. A small transition floor allows a path through disconnected candidates. Above a 20-second gap the search admits three extra graph hops.")
    doc.table(["Method","Top-1","Continuity","Wrong direction","Trips >=90%"],[[r["method"],pct(r["top1"]),pct(r["route_continuity"]),pct(r["wrong_direction_rate"]),pct(r["trips_ge90_share"])] for r in hmm],[152,75,90,107,91])
    doc.image(figs/"hmm_comparison.png","Student-test comparison from the saved decoder evaluation.",171)
    doc.para("The main classifier-HMM reached at least 90% point accuracy on eight of ten trips. Connectivity increased from 93.379% to 98.021%. Continuity alone cannot verify a correct road choice; a path can follow the wrong parallel road consistently.")
    changes=records(RESULTS/"decoder_changes.csv")
    gains=sum(int(row["corrected_by_hmm"]) for row in changes)
    losses=sum(int(row["lost_by_hmm"]) for row in changes)
    worst=max(changes,key=lambda r:int(r["lost_by_hmm"]))
    doc.para(f"Against the same aligned reference, Viterbi corrected {gains:,} wrong classifier choices but turned {losses:,} correct ones into errors. Trip {worst['trip']} had the most such regressions ({worst['lost_by_hmm']}). Sequence smoothing can therefore hide a correct local choice. These counts establish where the two outputs differ; they cannot establish the cause of each error.",size=9)
    doc.page("Deployment","Browser and API rehearsal")
    runs=records(ROOT/"report/rehearsal.csv")
    doc.para(f'The real browser selected each of {len(runs)} student-test trips and rendered {sum(int(r["n_points"]) for r in runs):,} observations in total. Both services ran on one host using separate HTTP origins. The rehearsal exercised health, trip discovery, POST preflight, prediction, route rendering and aggregate scoring.')
    doc.table(["Student trip","Observations","Accuracy","Continuity","Server ms"],[[r["trip_id"],r["n_points"],pct(r["accuracy"]),pct(r["route_continuity"]),f'{float(r["latency_ms"]):.0f}'] for r in runs],[110,110,100,100,95])
    doc.para("Laptop 1 runs scripts/start_api.ps1. Laptop 2 runs scripts/start_ui.ps1 with the API address on the shared LAN. The browser sends only the selected trip ID. The server performs prediction first, then reads its local reference to calculate the two scores.")
    if not evidence or not (evidence/"laptop1-terminal.png").is_file():
        doc.para("The physical two-laptop demonstration still requires matching UI and receiving-terminal screenshots from the same request. For a rehearsal, serve student_test; reserve eval_test for the evaluator. A local rehearsal cannot validate LAN reachability.")
    doc.para("Evidence: report/rehearsal.csv, report/rehearsal.log, results/model_comparison.csv, results/tuning_comparison.csv, results/hmm_results.csv, results/hmm_per_trip.csv and artifacts/best_params.json.",size=9)
    doc.image(ROOT/"report/screenshots/rehearsal-results.png","Browser result panel from the same local request for trip 00000007.",62)
    full=RESULTS/"all80"
    if (full/"tuning_comparison.csv").exists():
        trials=records(full/"optuna_trials.csv")
        full_models=records(full/"model_comparison.csv")
        full_tuning=records(full/"tuning_comparison.csv")
        summary=json.loads((full/"run.json").read_text())
        doc.page("Full training partition","Comparison across all 80 training trips")
        doc.para("This additional experiment includes all seven flagged trips. All five estimators use five-fold trip-grouped validation on the full training partition. The ten student-test trips are unchanged. This experiment is reported separately from the quality-filtered release demonstrated in the UI.")
        doc.table(["Model","CV F1","Top-1","Test F1","Continuity","Loss","ms/1k"],[[names[r["model"]],f'{float(r["cv_macro_f1"]):.3f}',pct(r["test_top1"]),f'{float(r["test_macro_f1"]):.3f}',pct(r["test_route_continuity"]),f'{float(r["test_log_loss"]):.3f}',f'{float(r["infer_ms_per_1k_rows"]):.2f}'] for r in full_models],[134,49,66,58,76,60,72])
        doc.model_scores(full_models,names)
        doc.para("Label quality changes the population on which each validation fold is scored. Differences between these runs reflect both learning on additional trips and measuring performance on a harder reference set.")
        diag=records(full/"fit_diagnostics.csv")
        doc.page("Full training partition","Training versus unseen trips")
        doc.para("Each training candidate was also scored by a model fitted on different trips. Comparing in-sample and pooled out-of-fold (OOF) scores on the same observations isolates the generalisation gap more directly than comparing distinct trip populations. Fold-mean and pooled scores weight trips differently.")
        doc.table(["Model","Train F1","OOF F1","Train Top-1","OOF Top-1","Test row acc."],[[names[r["model"]],pct(r["train_macro_f1"]),pct(r["pooled_oof_macro_f1"]),pct(r["train_top1"]),pct(r["pooled_oof_top1"]),pct(r["test_candidate_accuracy"])] for r in diag],[133,71,68,83,78,82])
        doc.table(["Model","Test precision","Test recall","Fit (s)","Five-fold CV (s)"],[[names[r["model"]],r["test_precision"],r["test_recall"],f'{float(r["fit_seconds"]):.1f}',f'{float(r["cv_seconds"]):.1f}'] for r in full_models],[142,105,85,77,106])
        gap=max(diag,key=lambda r:float(r["train_macro_f1"])-float(r["pooled_oof_macro_f1"]))
        doc.para(f'{names[gap["model"]]} had the largest fitted-to-OOF macro F1 gap ({100*(float(gap["train_macro_f1"])-float(gap["pooled_oof_macro_f1"])):.2f} percentage points). This is evidence of sensitivity to the training trips. No model in this comparison produces an epoch-wise validation-loss history.')
        doc.page("Full training partition","Bayesian search and refit")
        doc.para(f"The Bayesian search covered all 80 training trips in three grouped folds for each of {len(trials)} trials. The objective was mean observation-level Top-1. The best trial scored {pct(summary['search']['best_objective'])}; search time was {summary['search']['seconds']:.1f} s. The chosen hyperparameters were refitted and evaluated on the same partitions as the untuned configuration.")
        doc.table(["Configuration","CV F1","Test Top-1","Test F1","Loss","ms/1k"],[[r["configuration"],r["cv_macro_f1"],pct(r["test_top1"]),r["test_macro_f1"],r["test_log_loss"],r["infer_ms_per_1k_rows"]] for r in full_tuning],[134,74,83,74,74,76])
        doc.para("Full-partition winner by validation: "+summary["selected_file"]+". Search parameters: "+", ".join(f"{k}={v:.5g}" for k,v in summary["best_params"].items())+".")
        before,after=full_tuning
        doc.para(f'Student-test Top-1 changed by {100*(float(after["test_top1"])-float(before["test_top1"])):+.3f} percentage points after tuning. The choice between tuned and untuned was made on validation Top-1. The winning estimator was scored again through the existing HMM with its transition settings held fixed.')
        full_decode=records(full/"hmm_results.csv")
        doc.table(["All-80 checkpoint","Top-1","Continuity"],[[r["method"],pct(r["top1"]),pct(r["route_continuity"])] for r in full_decode if r["method"] in ("classifier_only(model)","hmm_classifier(model)")],[245,122,148])
        current=next(r for r in full_decode if r["method"]=="hmm_classifier(model)")
        screened=next(r for r in hmm if r["method"]=="hmm_classifier(model)")
        doc.para(f'Using the same decoder settings, student-test HMM Top-1 was {pct(current["top1"])} for the full-training checkpoint and {pct(screened["top1"])} for the quality-filtered release. The trials changed the training population and fitted weights together, so this comparison cannot isolate a single cause.',size=9)
        doc.image(full/"figures/optuna_history.png","TPE trial history on the full training partition.",155)
    if (full/"rehearsal.csv").exists():
        browser_runs=records(full/"rehearsal.csv")
        doc.para(f'The full-partition checkpoint also passed a live local browser-to-API rehearsal on {len(browser_runs)} trips and {sum(int(r["n_points"]) for r in browser_runs):,} observations, model version {browser_runs[0]["model_version"]}. Use -AllTrips on Laptop 1 for the full-training-partition demonstration.')
    if evidence:
        doc.page("Geographical inspection","Student-test comparison map")
        doc.image(evidence/"folium-submission.png","Folium view: grey road graph, blue GPS observations, green reference, dashed amber classifier and red HMM. Road geometry is supplied by the trip.",335)
        doc.image(figs/"pred_00000007.png","Projected trip 00000007. Outlined markers locate classifier/HMM disagreements; scores are calculated from arc IDs.",216)
        doc.page("Rehearsal evidence","A real browser prediction")
        doc.image(evidence/"ui-submission.png","Loopback rehearsal on student-test trip 00000007. The map and displayed scores came from the deployed classifier and HMM.",350)
        doc.para("Receiving-server log excerpt",True,13)
        for line in (ROOT/"report/rehearsal.log").read_text().splitlines():
            if "| 00000007 |" in line: doc.para(line,size=9)
    if evidence and (evidence/"ui-all80.png").exists():
        doc.page("Full training rehearsal","Browser result with the 80-trip model")
        doc.image(evidence/"ui-all80.png","Student-test rehearsal using the full-partition checkpoint on one host. The model version identifies this prediction.",350)
        doc.para("Matching receiving-server log entry",True,12)
        for line in (full/"rehearsal.log").read_text().splitlines():
            if "| 00000007 |" in line:
                doc.para(line,size=9)
    if evidence:
        ui, terminal = evidence/"laptop2-ui.png", evidence/"laptop1-terminal.png"
        if ui.is_file() != terminal.is_file():
            raise ValueError("Supply both screenshots from the same physical two-laptop request.")
        if ui.is_file():
            doc.page("Two-laptop demonstration","Matched physical-device evidence")
            doc.image(ui,"Laptop 2 browser result from the physical demonstration.",260)
            doc.image(terminal,"Laptop 1 receiving-terminal output for the same request.",240)
    doc.pdf.save()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"report/PathSnap_Report.pdf")
    parser.add_argument("--evidence",type=Path,help="Local directory containing Folium and evaluator rehearsal screenshots")
    args=parser.parse_args()
    build(args.output,args.evidence)

# Development data notes

- The archive contains 80 training trips and 10 student-test trips. The inventory counted 208,417 and 19,160 GPS observations, respectively.
- All 90 trips had their four expected files. The reference route is an arc sequence. The feature builder must align it to observations before creating candidate labels.
- The quality scan found no missing numeric values, invalid GPS coordinates, non-finite timestamps, or non-increasing time steps in either split.
- Using a numerical tolerance for one-second steps, 201,053 of 208,337 training intervals and 18,886 of 19,150 student-test intervals were one second.
- Gaps exceeded five seconds 1,216 times in training and 44 times in student testing. The maximum training gap was 153,064 seconds. The decoder should handle large gaps explicitly.
- There were multiple weakly connected graph components in 79 training trips and all 10 student-test trips. There were no isolated nodes under the incident-arc definition.
- Consecutive distinct reference arcs failed the directed endpoint check in 10 training trips and one student-test trip. The reference is useful for evaluation, with these discontinuities documented.
- Training nearest-road distances had a median of 4.97 m and a 95th percentile of 83.43 m. Student-test values were 4.76 m and 16.00 m. These distances use the nearest available segment. Candidate miss rate after reference alignment should guide radius and fallback decisions.
- Track step lengths use WGS84 geodesic distances. Nearest-segment distances use a per-trip UTM projection and full road segments.

Run `python -m visualisations.eda_plots` to regenerate the source tables and figures.

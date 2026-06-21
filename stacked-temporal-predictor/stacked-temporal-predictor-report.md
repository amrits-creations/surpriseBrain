What I saw from the execution of the stacked version is:
- The oddball tests:
    - The moving test performed SIGNIFICANTLY WORSE than the single layer temporal predictor. There was much less fluctuation in the overall surprise value though and it seemed to converge on a loss value before the oddball frames were introduced (this decreased the surprise value by a lot)
    - The static test also performed worse. The surprise value was a lot more than the single layer version.

- The live versiosn:
    - The learning capabilities were faster than the single layer version. However, it ddin't seem as good as the previous version at showing surprise. The surprise faded really fast and drastic changes in scenery did not affect the surprise map as much as I would have wanted it to.
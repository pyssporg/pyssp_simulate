package TestFMUs "Primitive FMU models for the pyssp_simulate test suite.

The package is self-contained: it has no Modelica Standard Library
dependency, so a bare OpenModelica installation can compile it. The
two models form the smallest possible co-simulation chain with an
analytic solution:

    RampSource:  y = t                     (unit ramp)
    Integrator:  der(y) = u, y(0) = 0      =>  y(t) = t^2 / 2

The signal connectors are short class definitions of the built-in
Real type, exactly like Modelica.Blocks.Interfaces.RealInput and
RealOutput. Keeping them Real subtypes is what makes this work with
plain variable names:

  * the flattened FMU variables are named ``y`` and ``u`` (not
    ``y.y`` / ``u.y`` as a long-form connector with a member would
    produce), so they match the connector names declared in the SSD
    and the signal names the backends record, and
  * the model equations can equate the connectors directly
    (``y = time``, ``der(y) = u``) without member access.
"

  connector RealInput = input Real "'input Real' as connector";

  connector RealOutput = output Real "'output Real' as connector";

  model RampSource "Ramp source: y = t"
    RealOutput y;
  equation
    y = time;
  end RampSource;

  model Integrator "Integrator: der(y) = u"
    RealInput u;
    RealOutput y(start = 0, fixed = true);
  equation
    der(y) = u;
  end Integrator;
end TestFMUs;

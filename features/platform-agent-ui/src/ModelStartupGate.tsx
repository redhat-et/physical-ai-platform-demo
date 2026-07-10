import React from "react";
import {
  Bullseye,
  Button,
  EmptyState,
  EmptyStateBody,
  EmptyStateFooter,
  EmptyStateActions,
  Spinner,
} from "@patternfly/react-core";
import { PlayIcon, ExclamationTriangleIcon } from "@patternfly/react-icons";

export type ModelGateState = "checking" | "not_ready" | "starting" | "error";

interface ModelStartupGateProps {
  state: ModelGateState;
  detail: string;
  slowWarning?: string;
  onStart: () => void;
  onRetry: () => void;
}

const ModelStartupGate: React.FC<ModelStartupGateProps> = ({
  state,
  detail,
  slowWarning,
  onStart,
  onRetry,
}) => {
  if (state === "checking") {
    return (
      <Bullseye>
        <Spinner size="lg" />
      </Bullseye>
    );
  }

  if (state === "not_ready") {
    return (
      <Bullseye>
        <EmptyState icon={PlayIcon} titleText="Model is not running">
          <EmptyStateBody>
            The agent&apos;s language model is scaled to zero to save resources.
            Start it to begin chatting — this can take a few minutes.
          </EmptyStateBody>
          <EmptyStateFooter>
            <EmptyStateActions>
              <Button variant="primary" onClick={onStart}>
                Start Model
              </Button>
            </EmptyStateActions>
          </EmptyStateFooter>
        </EmptyState>
      </Bullseye>
    );
  }

  if (state === "starting") {
    return (
      <Bullseye>
        <EmptyState icon={Spinner} titleText="Getting the model ready...">
          <EmptyStateBody>
            {detail}
            {slowWarning && (
              <>
                <br />
                {slowWarning}
              </>
            )}
          </EmptyStateBody>
        </EmptyState>
      </Bullseye>
    );
  }

  return (
    <Bullseye>
      <EmptyState
        icon={ExclamationTriangleIcon}
        titleText="Something went wrong"
        status="danger"
      >
        <EmptyStateBody>{detail}</EmptyStateBody>
        <EmptyStateFooter>
          <EmptyStateActions>
            <Button variant="primary" onClick={onRetry}>
              Try Again
            </Button>
          </EmptyStateActions>
        </EmptyStateFooter>
      </EmptyState>
    </Bullseye>
  );
};

export default ModelStartupGate;

import React from 'react';
import {
  EmptyState,
  EmptyStateBody,
  PageSection,
} from '@patternfly/react-core';
import { RobotIcon } from '@patternfly/react-icons';

const RoboticsPlayground: React.FC = () => (
  <PageSection>
    <EmptyState titleText="Robotics Playground" headingLevel="h1" icon={RobotIcon}>
      <EmptyStateBody>
        Experiment with robot policy models in a simulated environment.
        This feature is under development.
      </EmptyStateBody>
    </EmptyState>
  </PageSection>
);

export default RoboticsPlayground;
